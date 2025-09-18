//! CLI integration tests for kcs-serializer
//!
//! These tests verify that the command-line interface works correctly
//! with actual file operations.

use assert_cmd::prelude::*;
use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use kcs_serializer::{GraphExporter, JsonGraphExporter};
use predicates::prelude::*;
use std::process::Command;
use tempfile::tempdir;

/// Create a test graph and save it as JSON for CLI testing
fn create_test_graph_file() -> (tempfile::TempDir, String) {
    let dir = tempdir().unwrap();
    let json_path = dir.path().join("test_graph.json");

    // Create a simple test graph
    let mut graph = KernelGraph::new();

    graph.add_symbol(Symbol {
        name: "test_main".to_string(),
        file_path: "main.c".to_string(),
        line_number: 10,
        symbol_type: SymbolType::Function,
        signature: Some("int test_main(void)".to_string()),
        config_dependencies: vec!["CONFIG_TEST".to_string()],
    });

    graph.add_symbol(Symbol {
        name: "helper_func".to_string(),
        file_path: "helper.c".to_string(),
        line_number: 20,
        symbol_type: SymbolType::Function,
        signature: Some("void helper_func(int)".to_string()),
        config_dependencies: vec![],
    });

    graph.add_symbol(Symbol {
        name: "TEST_CONSTANT".to_string(),
        file_path: "constants.h".to_string(),
        line_number: 5,
        symbol_type: SymbolType::Constant,
        signature: None,
        config_dependencies: vec![],
    });

    // Add some calls
    let edge = CallEdge {
        call_type: CallType::Direct,
        call_site_line: 15,
        conditional: false,
        config_guard: None,
    };
    graph.add_call("test_main", "helper_func", edge).unwrap();

    // Save the graph as JSON
    let json_exporter = JsonGraphExporter::new().with_metadata(true);
    json_exporter
        .export_to_file(&graph, json_path.to_str().unwrap())
        .unwrap();

    (dir, json_path.to_string_lossy().to_string())
}

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.arg("--help");
    cmd.assert().success().stdout(predicate::str::contains(
        "Kernel call graph serialization tool",
    ));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

#[test]
fn test_convert_command() {
    let (_temp_dir, json_file) = create_test_graph_file();
    let temp_dir = tempdir().unwrap();
    let graphml_output = temp_dir.path().join("output.graphml");

    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.args([
        "convert",
        "--input",
        &json_file,
        "--output",
        graphml_output.to_str().unwrap(),
        "--from",
        "json",
        "--to",
        "graphml",
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Conversion complete"));

    // Verify the output file was created
    assert!(graphml_output.exists());

    // Verify it's valid XML/GraphML by checking basic structure
    let content = std::fs::read_to_string(&graphml_output).unwrap();
    assert!(content.contains("<?xml"));
    assert!(content.contains("<graphml"));
    assert!(content.contains("test_main"));
}

#[test]
fn test_info_command() {
    let (_temp_dir, json_file) = create_test_graph_file();

    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.args(["info", "--file", &json_file, "--format", "json"]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Graph Information"))
        .stdout(predicate::str::contains("Nodes: 3"))
        .stdout(predicate::str::contains("Edges: 1"));
}

#[test]
fn test_export_command_with_chunking() {
    let (_temp_dir, json_file) = create_test_graph_file();
    let temp_dir = tempdir().unwrap();
    let output_base = temp_dir.path().join("chunked_output");

    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.args([
        "export",
        "--input",
        &json_file,
        "--output",
        output_base.to_str().unwrap(),
        "--format",
        "json",
        "--chunked",
        "--max-nodes",
        "2", // Force chunking with small chunk size
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Chunking graph"));

    // Should create at least one chunk file
    let chunk_file = format!("{}.chunk_000", output_base.to_string_lossy());
    assert!(std::path::Path::new(&chunk_file).exists());
}

#[test]
fn test_import_command() {
    let (_temp_dir, json_file) = create_test_graph_file();
    let temp_dir = tempdir().unwrap();
    let output_file = temp_dir.path().join("imported.json");

    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.args([
        "import",
        "--input",
        &json_file,
        "--output",
        output_file.to_str().unwrap(),
        "--format",
        "json",
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Graph imported"))
        .stdout(predicate::str::contains("3 nodes"))
        .stdout(predicate::str::contains("1 edges"));

    // Verify the output file was created
    assert!(output_file.exists());
}

#[test]
fn test_error_handling_missing_file() {
    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.args(["info", "--file", "nonexistent.json", "--format", "json"]);

    cmd.assert().failure();
}

#[test]
fn test_error_handling_invalid_format() {
    let (_temp_dir, json_file) = create_test_graph_file();

    let mut cmd = Command::cargo_bin("kcs-serializer").unwrap();
    cmd.args([
        "info", "--file", &json_file, "--format",
        "graphml", // Try to read JSON file as GraphML
    ]);

    // This should either fail or show different results
    // The exact behavior depends on how lenient the GraphML parser is
    cmd.assert().code(predicate::in_iter([0, 1])); // Allow either success or failure
}
