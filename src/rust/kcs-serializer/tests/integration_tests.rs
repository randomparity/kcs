//! Integration tests for kcs-serializer
//!
//! These tests verify the interaction between different serialization formats
//! and test the complete serialization pipeline.

use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use kcs_serializer::{
    ChunkOptions, GraphChunk, GraphChunker, GraphExporter, GraphImporter, GraphMLExporter,
    JsonGraphExporter,
};
use tempfile::tempdir;

/// Create a complex test graph with various symbol types and call relationships
fn create_complex_test_graph() -> KernelGraph {
    let mut graph = KernelGraph::new();

    // Add different types of symbols
    let symbols = vec![
        Symbol {
            name: "kernel_init".to_string(),
            file_path: "init/main.c".to_string(),
            line_number: 100,
            symbol_type: SymbolType::Function,
            signature: Some("asmlinkage __visible void __init start_kernel(void)".to_string()),
            config_dependencies: vec!["CONFIG_PRINTK".to_string()],
        },
        Symbol {
            name: "EXPORT_SYMBOL".to_string(),
            file_path: "include/linux/export.h".to_string(),
            line_number: 50,
            symbol_type: SymbolType::Macro,
            signature: None,
            config_dependencies: vec!["CONFIG_MODULES".to_string()],
        },
        Symbol {
            name: "task_struct".to_string(),
            file_path: "include/linux/sched.h".to_string(),
            line_number: 800,
            symbol_type: SymbolType::Type,
            signature: None,
            config_dependencies: vec![],
        },
        Symbol {
            name: "current".to_string(),
            file_path: "arch/x86/include/asm/current.h".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Variable,
            signature: Some("DECLARE_PER_CPU(struct task_struct *, current_task)".to_string()),
            config_dependencies: vec!["CONFIG_SMP".to_string()],
        },
        Symbol {
            name: "PAGE_SIZE".to_string(),
            file_path: "arch/x86/include/asm/page_types.h".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Constant,
            signature: None,
            config_dependencies: vec![],
        },
        Symbol {
            name: "printk".to_string(),
            file_path: "kernel/printk/printk.c".to_string(),
            line_number: 1500,
            symbol_type: SymbolType::Function,
            signature: Some("__printf(1, 2) int printk(const char *fmt, ...)".to_string()),
            config_dependencies: vec!["CONFIG_PRINTK".to_string()],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Add various types of call relationships
    let calls = vec![
        ("kernel_init", "printk", CallType::Direct, 105, false, None),
        (
            "kernel_init",
            "EXPORT_SYMBOL",
            CallType::Macro,
            200,
            true,
            Some("CONFIG_MODULES".to_string()),
        ),
        ("printk", "current", CallType::Indirect, 1520, false, None),
        (
            "kernel_init",
            "task_struct",
            CallType::Direct,
            150,
            true,
            Some("CONFIG_SMP".to_string()),
        ),
        ("printk", "PAGE_SIZE", CallType::Direct, 1600, false, None),
    ];

    for (caller, callee, call_type, line, conditional, config_guard) in calls {
        let edge = CallEdge {
            call_type,
            call_site_line: line,
            conditional,
            config_guard,
        };
        graph.add_call(caller, callee, edge).unwrap();
    }

    graph
}

/// Create a large test graph for performance testing
fn create_large_test_graph(node_count: usize) -> KernelGraph {
    let mut graph = KernelGraph::new();

    // Create many symbols
    for i in 0..node_count {
        let symbol = Symbol {
            name: format!("func_{}", i),
            file_path: format!("file_{}.c", i / 100),
            line_number: (i % 1000) as u32 + 1,
            symbol_type: match i % 4 {
                0 => SymbolType::Function,
                1 => SymbolType::Variable,
                2 => SymbolType::Macro,
                _ => SymbolType::Type,
            },
            signature: if i % 3 == 0 {
                Some(format!("signature_{}", i))
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

    // Create call relationships in a pattern
    for i in 0..node_count - 1 {
        if i % 3 == 0 {
            let edge = CallEdge {
                call_type: match i % 4 {
                    0 => CallType::Direct,
                    1 => CallType::Indirect,
                    2 => CallType::FunctionPointer,
                    _ => CallType::Macro,
                },
                call_site_line: (i % 1000) as u32 + 1,
                conditional: i % 7 == 0,
                config_guard: if i % 11 == 0 {
                    Some(format!("CONFIG_GUARD_{}", i))
                } else {
                    None
                },
            };
            graph
                .add_call(&format!("func_{}", i), &format!("func_{}", i + 1), edge)
                .unwrap();
        }
    }

    graph
}

#[test]
fn test_json_graphml_format_compatibility() {
    let graph = create_complex_test_graph();

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Export to both formats
    let json_string = json_exporter.export_to_string(&graph).unwrap();
    let graphml_string = graphml_exporter.export_to_string(&graph).unwrap();

    // Import from both formats
    let json_imported = json_exporter.import_from_string(&json_string).unwrap();
    let graphml_imported = graphml_exporter
        .import_from_string(&graphml_string)
        .unwrap();

    // Both imported graphs should have the same structure
    assert_eq!(
        json_imported.symbol_count(),
        graphml_imported.symbol_count()
    );
    assert_eq!(json_imported.symbol_count(), graph.symbol_count());

    // Check that all symbols exist in both imported graphs
    let original_symbols = [
        "kernel_init",
        "EXPORT_SYMBOL",
        "task_struct",
        "current",
        "PAGE_SIZE",
        "printk",
    ];
    for symbol_name in &original_symbols {
        assert!(json_imported.get_symbol(symbol_name).is_some());
        assert!(graphml_imported.get_symbol(symbol_name).is_some());
    }
}

#[test]
fn test_cross_format_serialization() {
    let graph = create_complex_test_graph();

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Export with JSON, import with GraphML should fail gracefully
    let json_string = json_exporter.export_to_string(&graph).unwrap();
    let graphml_import_result = graphml_exporter.import_from_string(&json_string);
    // GraphML parser might be lenient but should produce different results
    if let Ok(imported_graph) = graphml_import_result {
        // If it succeeds, it should produce an empty or very different graph
        assert!(
            imported_graph.symbol_count() != graph.symbol_count()
                || imported_graph.symbol_count() == 0
        );
    }

    // Export with GraphML, import with JSON should fail gracefully
    let graphml_string = graphml_exporter.export_to_string(&graph).unwrap();
    let json_import_result = json_exporter.import_from_string(&graphml_string);
    assert!(json_import_result.is_err());
}

#[test]
fn test_file_serialization_roundtrip() {
    let graph = create_complex_test_graph();
    let dir = tempdir().unwrap();

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Test JSON file roundtrip
    let json_path = dir.path().join("test_graph.json");
    json_exporter
        .export_to_file(&graph, json_path.to_str().unwrap())
        .unwrap();
    let json_imported = json_exporter
        .import_from_file(json_path.to_str().unwrap())
        .unwrap();
    assert_eq!(json_imported.symbol_count(), graph.symbol_count());

    // Test GraphML file roundtrip
    let graphml_path = dir.path().join("test_graph.graphml");
    graphml_exporter
        .export_to_file(&graph, graphml_path.to_str().unwrap())
        .unwrap();
    let graphml_imported = graphml_exporter
        .import_from_file(graphml_path.to_str().unwrap())
        .unwrap();
    assert_eq!(graphml_imported.symbol_count(), graph.symbol_count());
}

#[test]
fn test_chunking_with_serialization() {
    let graph = create_complex_test_graph();

    let chunker_options = ChunkOptions {
        max_nodes_per_chunk: 3,
        max_edges_per_chunk: 3,
        preserve_components: false,
        include_boundary_edges: true,
    };
    let chunker = GraphChunker::with_options(chunker_options);
    let _json_exporter = JsonGraphExporter::new();

    // Chunk the graph
    let chunks = chunker.chunk_graph(&graph).unwrap();
    assert!(chunks.len() >= 2);

    // Serialize each chunk as JSON and verify it can be parsed
    for chunk in &chunks {
        let chunk_json = serde_json::to_string(&chunk).unwrap();
        let parsed_chunk: GraphChunk = serde_json::from_str(&chunk_json).unwrap();
        assert_eq!(parsed_chunk.id, chunk.id);
        assert_eq!(parsed_chunk.node_ids.len(), chunk.node_ids.len());
    }

    // Reassemble and verify
    let reassembled = chunker.reassemble_chunks(&chunks).unwrap();
    assert_eq!(reassembled.symbol_count(), graph.symbol_count());
}

#[test]
fn test_large_graph_serialization_performance() {
    let graph = create_large_test_graph(100); // 100 nodes for reasonable test time

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Test JSON serialization performance
    let start = std::time::Instant::now();
    let json_string = json_exporter.export_to_string(&graph).unwrap();
    let json_export_time = start.elapsed();

    let start = std::time::Instant::now();
    let _json_imported = json_exporter.import_from_string(&json_string).unwrap();
    let json_import_time = start.elapsed();

    // Test GraphML serialization performance
    let start = std::time::Instant::now();
    let graphml_string = graphml_exporter.export_to_string(&graph).unwrap();
    let graphml_export_time = start.elapsed();

    let start = std::time::Instant::now();
    let _graphml_imported = graphml_exporter
        .import_from_string(&graphml_string)
        .unwrap();
    let graphml_import_time = start.elapsed();

    // Performance should be reasonable (less than 1 second for 100 nodes)
    assert!(json_export_time.as_millis() < 1000);
    assert!(json_import_time.as_millis() < 1000);
    assert!(graphml_export_time.as_millis() < 1000);
    assert!(graphml_import_time.as_millis() < 1000);

    // JSON should generally be faster than GraphML
    println!(
        "JSON export: {:?}, import: {:?}",
        json_export_time, json_import_time
    );
    println!(
        "GraphML export: {:?}, import: {:?}",
        graphml_export_time, graphml_import_time
    );
}

#[test]
fn test_memory_usage_chunking() {
    let graph = create_large_test_graph(200);

    let chunker_options = ChunkOptions {
        max_nodes_per_chunk: 20,
        max_edges_per_chunk: 30,
        preserve_components: false,
        include_boundary_edges: true,
    };
    let chunker = GraphChunker::with_options(chunker_options);

    // Test streaming vs direct chunking
    let direct_chunks = chunker.chunk_graph(&graph).unwrap();

    let streaming_chunks: Result<Vec<_>, _> = chunker.stream_chunks(&graph).collect();
    let streaming_chunks = streaming_chunks.unwrap();

    assert_eq!(direct_chunks.len(), streaming_chunks.len());

    // Verify chunk constraints are respected
    for chunk in &direct_chunks {
        assert!(chunk.node_ids.len() <= 20);
    }
}

#[test]
fn test_empty_and_single_node_graphs() {
    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Test empty graph
    let empty_graph = KernelGraph::new();
    let json_empty = json_exporter.export_to_string(&empty_graph).unwrap();
    let graphml_empty = graphml_exporter.export_to_string(&empty_graph).unwrap();

    let json_imported_empty = json_exporter.import_from_string(&json_empty).unwrap();
    let graphml_imported_empty = graphml_exporter.import_from_string(&graphml_empty).unwrap();

    assert_eq!(json_imported_empty.symbol_count(), 0);
    assert_eq!(graphml_imported_empty.symbol_count(), 0);

    // Test single node graph
    let mut single_graph = KernelGraph::new();
    single_graph.add_symbol(Symbol {
        name: "single_func".to_string(),
        file_path: "test.c".to_string(),
        line_number: 1,
        symbol_type: SymbolType::Function,
        signature: None,
        config_dependencies: vec![],
    });

    let json_single = json_exporter.export_to_string(&single_graph).unwrap();
    let graphml_single = graphml_exporter.export_to_string(&single_graph).unwrap();

    let json_imported_single = json_exporter.import_from_string(&json_single).unwrap();
    let graphml_imported_single = graphml_exporter
        .import_from_string(&graphml_single)
        .unwrap();

    assert_eq!(json_imported_single.symbol_count(), 1);
    assert_eq!(graphml_imported_single.symbol_count(), 1);
    assert!(json_imported_single.get_symbol("single_func").is_some());
    assert!(graphml_imported_single.get_symbol("single_func").is_some());
}

#[test]
fn test_serialization_with_unicode_and_special_chars() {
    let mut graph = KernelGraph::new();

    // Add symbols with Unicode and special characters
    let symbols = vec![
        Symbol {
            name: "测试函数".to_string(), // Chinese characters
            file_path: "unicode_file.c".to_string(),
            line_number: 1,
            symbol_type: SymbolType::Function,
            signature: Some("int 测试函数(void)".to_string()),
            config_dependencies: vec!["CONFIG_UTF8".to_string()],
        },
        Symbol {
            name: "func_with_<>&\"'_chars".to_string(), // XML special chars
            file_path: "special&file.c".to_string(),
            line_number: 2,
            symbol_type: SymbolType::Function,
            signature: Some("void func<T>(T& value)".to_string()),
            config_dependencies: vec!["CONFIG_TEMPLATE<T>".to_string()],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Test JSON with Unicode
    let json_string = json_exporter.export_to_string(&graph).unwrap();
    let json_imported = json_exporter.import_from_string(&json_string).unwrap();
    assert!(json_imported.get_symbol("测试函数").is_some());
    assert!(json_imported.get_symbol("func_with_<>&\"'_chars").is_some());

    // Test GraphML with Unicode and special chars
    let graphml_string = graphml_exporter.export_to_string(&graph).unwrap();
    let graphml_imported = graphml_exporter
        .import_from_string(&graphml_string)
        .unwrap();
    assert!(graphml_imported.get_symbol("测试函数").is_some());
    assert!(graphml_imported
        .get_symbol("func_with_<>&\"'_chars")
        .is_some());
}

#[test]
fn test_metadata_preservation() {
    let graph = create_complex_test_graph();

    // Test JSON metadata
    let json_exporter = JsonGraphExporter::new().with_metadata(true);
    let json_string = json_exporter.export_to_string(&graph).unwrap();

    // Parse as JSON to verify metadata structure
    let json_value: serde_json::Value = serde_json::from_str(&json_string).unwrap();
    assert!(json_value.get("metadata").is_some());
    let metadata = json_value.get("metadata").unwrap();
    assert!(metadata.get("format_version").is_some());
    assert!(metadata.get("created_at").is_some());
    assert!(metadata.get("created_by").is_some());
    assert_eq!(
        metadata.get("node_count").unwrap().as_u64().unwrap() as usize,
        graph.symbol_count()
    );

    // Test without metadata
    let json_exporter_no_meta = JsonGraphExporter::new().with_metadata(false);
    let json_string_no_meta = json_exporter_no_meta.export_to_string(&graph).unwrap();
    let json_value_no_meta: serde_json::Value = serde_json::from_str(&json_string_no_meta).unwrap();
    assert!(json_value_no_meta.get("metadata").is_none());
}

#[test]
fn test_format_specific_features() {
    let graph = create_complex_test_graph();

    // Test JSON specific features
    let json_pretty = JsonGraphExporter::new().with_pretty(true);
    let json_compact = JsonGraphExporter::new().with_pretty(false);

    let pretty_string = json_pretty.export_to_string(&graph).unwrap();
    let compact_string = json_compact.export_to_string(&graph).unwrap();

    // Pretty format should have newlines, compact shouldn't
    assert!(pretty_string.contains('\n'));
    assert!(!compact_string.contains('\n'));

    // Both should import to the same graph
    let pretty_imported = json_pretty.import_from_string(&pretty_string).unwrap();
    let compact_imported = json_compact.import_from_string(&compact_string).unwrap();
    assert_eq!(
        pretty_imported.symbol_count(),
        compact_imported.symbol_count()
    );

    // Test GraphML specific features
    let graphml_with_attrs = GraphMLExporter::new().with_attributes(true);
    let graphml_no_attrs = GraphMLExporter::new().with_attributes(false);

    let attrs_string = graphml_with_attrs.export_to_string(&graph).unwrap();
    let no_attrs_string = graphml_no_attrs.export_to_string(&graph).unwrap();

    // With attributes should be longer and contain data elements
    assert!(attrs_string.len() > no_attrs_string.len());
    assert!(attrs_string.contains("<data key=\"d0\">"));
    assert!(!no_attrs_string.contains("<data key=\"d0\">"));
}
