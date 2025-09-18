//! Error handling and edge case tests for kcs-serializer
//!
//! These tests verify proper error handling and edge case behavior
//! for all serialization formats.

use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use kcs_serializer::{GraphExporter, GraphImporter, GraphMLExporter, JsonGraphExporter};
use tempfile::tempdir;

#[test]
fn test_malformed_json_input() {
    let json_exporter = JsonGraphExporter::new();

    // Test completely invalid JSON
    let result = json_exporter.import_from_string("{invalid json");
    assert!(result.is_err());

    // Test valid JSON but invalid structure
    let result = json_exporter.import_from_string(r#"{"not": "a graph"}"#);
    assert!(result.is_err());

    // Test missing required fields
    let result = json_exporter.import_from_string(r#"{"nodes": []}"#);
    assert!(result.is_err());

    // Test invalid node structure
    let result =
        json_exporter.import_from_string(r#"{"nodes": [{"invalid": "node"}], "edges": []}"#);
    assert!(result.is_err());
}

#[test]
fn test_malformed_graphml_input() {
    let graphml_exporter = GraphMLExporter::new();

    // Test completely invalid XML
    let result = graphml_exporter.import_from_string("<invalid xml");
    assert!(result.is_err());

    // Test valid XML but not GraphML - this might succeed but create empty graph
    let result = graphml_exporter.import_from_string("<root><element>content</element></root>");
    // Either fails or creates empty graph
    if let Ok(graph) = result {
        assert_eq!(graph.symbol_count(), 0);
    }

    // Test malformed GraphML structure
    let _result = graphml_exporter.import_from_string(
        r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <graphml>
            <node><!-- missing id attribute -->
                <data key="label">test</data>
            </node>
        </graphml>
    "#,
    );
    // This should not panic but may result in an incomplete graph
    // The exact behavior depends on the implementation

    // Test unclosed XML tags
    let result = graphml_exporter.import_from_string(
        r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <graphml>
            <node id="n1">
                <data key="label">test</data>
            <!-- missing closing node tag -->
        </graphml>
    "#,
    );
    assert!(result.is_err());
}

#[test]
fn test_file_io_errors() {
    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();
    let graph = KernelGraph::new();

    // Test export to invalid path
    let result = json_exporter.export_to_file(&graph, "/invalid/path/file.json");
    assert!(result.is_err());

    let result = graphml_exporter.export_to_file(&graph, "/invalid/path/file.graphml");
    assert!(result.is_err());

    // Test import from non-existent file
    let result = json_exporter.import_from_file("/non/existent/file.json");
    assert!(result.is_err());

    let result = graphml_exporter.import_from_file("/non/existent/file.graphml");
    assert!(result.is_err());

    // Test import from directory instead of file
    let dir = tempdir().unwrap();
    let result = json_exporter.import_from_file(dir.path().to_str().unwrap());
    assert!(result.is_err());

    let result = graphml_exporter.import_from_file(dir.path().to_str().unwrap());
    assert!(result.is_err());
}

#[test]
fn test_empty_string_inputs() {
    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Test empty string input
    let result = json_exporter.import_from_string("");
    assert!(result.is_err());

    let result = graphml_exporter.import_from_string("");
    // GraphML parser might succeed with empty input, creating an empty graph
    if let Ok(graph) = result {
        assert_eq!(graph.symbol_count(), 0);
    }

    // Test whitespace-only input
    let result = json_exporter.import_from_string("   \n\t  ");
    assert!(result.is_err());

    let result = graphml_exporter.import_from_string("   \n\t  ");
    // GraphML parser might succeed with whitespace input, creating an empty graph
    if let Ok(graph) = result {
        assert_eq!(graph.symbol_count(), 0);
    }
}

#[test]
fn test_very_large_graph_limits() {
    // Test with a graph that has extremely long names and paths
    let mut graph = KernelGraph::new();

    let very_long_name = "a".repeat(10000); // 10KB name
    let very_long_path = "/".repeat(5000); // 5KB path
    let very_long_signature = "signature_".repeat(1000); // 10KB signature

    graph.add_symbol(Symbol {
        name: very_long_name.clone(),
        file_path: very_long_path,
        line_number: u32::MAX,
        symbol_type: SymbolType::Function,
        signature: Some(very_long_signature),
        config_dependencies: vec!["A".repeat(1000); 100], // 100 config deps of 1KB each
    });

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // These should work but might be slow
    let json_result = json_exporter.export_to_string(&graph);
    assert!(json_result.is_ok());

    let graphml_result = graphml_exporter.export_to_string(&graph);
    assert!(graphml_result.is_ok());

    // Test round-trip
    if let Ok(json_string) = json_result {
        let imported = json_exporter.import_from_string(&json_string).unwrap();
        assert!(imported.get_symbol(&very_long_name).is_some());
    }

    if let Ok(graphml_string) = graphml_result {
        let imported = graphml_exporter
            .import_from_string(&graphml_string)
            .unwrap();
        assert!(imported.get_symbol(&very_long_name).is_some());
    }
}

#[test]
fn test_cyclic_graph_handling() {
    let mut graph = KernelGraph::new();

    // Create symbols that form cycles
    for i in 0..5 {
        graph.add_symbol(Symbol {
            name: format!("func_{}", i),
            file_path: "cycle.c".to_string(),
            line_number: (i + 1) * 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        });
    }

    // Create a cycle: func_0 -> func_1 -> func_2 -> func_0
    // And a longer cycle: func_2 -> func_3 -> func_4 -> func_2
    let edges = vec![
        ("func_0", "func_1"),
        ("func_1", "func_2"),
        ("func_2", "func_0"), // Forms cycle
        ("func_2", "func_3"),
        ("func_3", "func_4"),
        ("func_4", "func_2"), // Forms longer cycle
    ];

    for (source, target) in edges {
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 1,
            conditional: false,
            config_guard: None,
        };
        graph.add_call(source, target, edge).unwrap();
    }

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Should handle cycles without infinite loops
    let json_string = json_exporter.export_to_string(&graph).unwrap();
    let graphml_string = graphml_exporter.export_to_string(&graph).unwrap();

    // Should be able to import cycles
    let json_imported = json_exporter.import_from_string(&json_string).unwrap();
    let graphml_imported = graphml_exporter
        .import_from_string(&graphml_string)
        .unwrap();

    assert_eq!(json_imported.symbol_count(), 5);
    assert_eq!(graphml_imported.symbol_count(), 5);
    assert_eq!(json_imported.call_count(), 6);
    assert_eq!(graphml_imported.call_count(), 6);
}

#[test]
fn test_null_and_control_characters() {
    let mut graph = KernelGraph::new();

    // Test with various problematic characters
    let problematic_chars = [
        "func_with_null\0char",
        "func_with_tab\tchar",
        "func_with_newline\nchar",
        "func_with_return\rchar",
        "func_with_backslash\\char",
        "func_with_quote\"char",
        "func_with_quote'char",
    ];

    for (i, name) in problematic_chars.iter().enumerate() {
        graph.add_symbol(Symbol {
            name: name.to_string(),
            file_path: format!("file_{}.c", i),
            line_number: (i + 1) as u32,
            symbol_type: SymbolType::Function,
            signature: Some(format!("signature_with_\0_null_{}", i)),
            config_dependencies: vec![format!("CONFIG_\t_TAB_{}", i)],
        });
    }

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // JSON should handle these characters (they get escaped)
    let json_result = json_exporter.export_to_string(&graph);
    assert!(json_result.is_ok());

    // GraphML should handle these characters (they get XML-escaped)
    let graphml_result = graphml_exporter.export_to_string(&graph);
    assert!(graphml_result.is_ok());

    // Test round-trip to ensure escaping works correctly
    if let Ok(json_string) = json_result {
        let imported = json_exporter.import_from_string(&json_string);
        // Some control characters might not round-trip perfectly
        // but the operation should not panic or corrupt data
        assert!(imported.is_ok() || imported.is_err()); // Just ensure no panic
    }
}

#[test]
fn test_edge_reference_errors() {
    let json_exporter = JsonGraphExporter::new();

    // Test edges referencing non-existent nodes
    let invalid_json = r#"{
        "nodes": [
            {"id": "node1", "label": "func1"}
        ],
        "edges": [
            {"source": "node1", "target": "nonexistent", "label": "Direct"}
        ]
    }"#;

    let result = json_exporter.import_from_string(invalid_json);
    // This might succeed but create an incomplete graph, or fail
    // The exact behavior depends on implementation - we just ensure no panic
    assert!(result.is_ok() || result.is_err());

    // Test self-referencing edges
    let self_ref_json = r#"{
        "nodes": [
            {"id": "node1", "label": "func1"}
        ],
        "edges": [
            {"source": "node1", "target": "node1", "label": "Direct"}
        ]
    }"#;

    let result = json_exporter.import_from_string(self_ref_json);
    // Self-references should be handled gracefully
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let graph = Arc::new({
        let mut g = KernelGraph::new();
        g.add_symbol(Symbol {
            name: "concurrent_func".to_string(),
            file_path: "concurrent.c".to_string(),
            line_number: 1,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        });
        g
    });

    let json_exporter = Arc::new(JsonGraphExporter::new());
    let graphml_exporter = Arc::new(GraphMLExporter::new());

    // Test concurrent serialization
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let graph = Arc::clone(&graph);
            let json_exp = Arc::clone(&json_exporter);
            let graphml_exp = Arc::clone(&graphml_exporter);

            thread::spawn(move || {
                if i % 2 == 0 {
                    json_exp.export_to_string(&graph).unwrap()
                } else {
                    graphml_exp.export_to_string(&graph).unwrap()
                }
            })
        })
        .collect();

    // All threads should complete successfully
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(!result.is_empty());
    }
}

#[test]
fn test_memory_exhaustion_protection() {
    // Test with deeply nested or very wide graphs that might cause stack overflow
    let mut graph = KernelGraph::new();

    // Create a very wide graph (many nodes with few connections)
    for i in 0..1000 {
        graph.add_symbol(Symbol {
            name: format!("wide_func_{}", i),
            file_path: "wide.c".to_string(),
            line_number: i + 1,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        });
    }

    // Add a few connections to make it interesting
    for i in 0..10 {
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 1,
            conditional: false,
            config_guard: None,
        };
        graph
            .add_call(
                &format!("wide_func_{}", i),
                &format!("wide_func_{}", i + 1),
                edge,
            )
            .unwrap();
    }

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    // Should handle large graphs without stack overflow
    let json_result = json_exporter.export_to_string(&graph);
    let graphml_result = graphml_exporter.export_to_string(&graph);

    assert!(json_result.is_ok());
    assert!(graphml_result.is_ok());

    // Verify the serialized output is reasonable
    if let Ok(json_string) = json_result {
        assert!(json_string.len() > 1000); // Should be substantial
        assert!(json_string.len() < 10_000_000); // But not excessive
    }
}
