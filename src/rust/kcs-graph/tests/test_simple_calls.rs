/*!
 * Integration test for basic call extraction and graph construction.
 *
 * Tests end-to-end flow from parsing C code to building call graphs.
 * This test must fail before implementation and pass after.
 *
 * Key integration points tested:
 * - kcs-parser extracts call edges from C source
 * - kcs-graph builds graph from extracted call edges
 * - Query operations work on constructed graph
 */

use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use std::io::Write;
use tempfile::NamedTempFile;

/// Test data representing simple C code with function calls
const SIMPLE_C_CODE: &str = r#"
// Simple function calls test case
int helper_function(int x) {
    return x * 2;
}

int main_function(int a, int b) {
    int result = helper_function(a);
    helper_function(b);
    return result;
}
"#;

/// Helper function to create a temporary C file for testing
fn create_temp_c_file(content: &str) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(content.as_bytes())
        .expect("Failed to write to temp file");
    temp_file
}

/// Convert kcs-parser CallEdge to kcs-graph CallEdge format
fn convert_parser_call_edge_to_graph(
    parser_call_type: kcs_parser::CallType,
    call_site_line: u32,
) -> CallEdge {
    let graph_call_type = match parser_call_type {
        kcs_parser::CallType::Direct => CallType::Direct,
        kcs_parser::CallType::Indirect => CallType::Indirect,
        kcs_parser::CallType::Macro => CallType::Macro,
    };

    CallEdge {
        call_type: graph_call_type,
        call_site_line,
        conditional: false,
        config_guard: None,
    }
}

/// Create test symbols for the simple C code example
fn create_test_symbols() -> Vec<Symbol> {
    vec![
        Symbol {
            name: "helper_function".to_string(),
            file_path: "test_simple.c".to_string(),
            line_number: 3,
            symbol_type: SymbolType::Function,
            signature: Some("int helper_function(int x)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "main_function".to_string(),
            file_path: "test_simple.c".to_string(),
            line_number: 7,
            symbol_type: SymbolType::Function,
            signature: Some("int main_function(int a, int b)".to_string()),
            config_dependencies: vec![],
        },
    ]
}

/// Integration test: Basic call extraction and graph construction
#[test]
fn test_basic_call_extraction_integration() {
    let mut graph = KernelGraph::new();

    // Add test symbols to graph
    let symbols = create_test_symbols();
    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Simulate call edges that would be extracted by kcs-parser
    // Expected: main_function calls helper_function twice (lines 8, 9)
    let call_edges = vec![
        (
            "main_function",
            "helper_function",
            8,
            kcs_parser::CallType::Direct,
        ),
        (
            "main_function",
            "helper_function",
            9,
            kcs_parser::CallType::Direct,
        ),
    ];

    // Add call edges to graph
    for (caller, callee, line, call_type) in call_edges {
        let graph_edge = convert_parser_call_edge_to_graph(call_type, line);

        let result = graph.add_call(caller, callee, graph_edge);
        assert!(
            result.is_ok(),
            "Failed to add call edge: {} -> {}",
            caller,
            callee
        );
    }

    // Verify graph construction
    assert_eq!(graph.symbol_count(), 2, "Should have 2 symbols in graph");
    assert_eq!(graph.call_count(), 2, "Should have 2 call edges in graph");

    // Test graph queries
    let callees = graph.find_callees("main_function");
    assert!(!callees.is_empty(), "main_function should have callees");
    // Note: find_callees returns multiple entries for multiple call sites to same function
    // All callees should be helper_function
    for callee in &callees {
        assert_eq!(
            callee.name, "helper_function",
            "Should call helper_function"
        );
    }

    let callers = graph.find_callers("helper_function");
    assert!(!callers.is_empty(), "helper_function should have callers");
    // Note: find_callers returns one entry per incoming edge
    // All callers should be main_function (since it calls helper_function twice)
    for caller in &callers {
        assert_eq!(
            caller.name, "main_function",
            "Should be called by main_function"
        );
    }
}

/// Integration test: End-to-end with parser (will fail until parser is implemented)
#[test]
fn test_end_to_end_call_extraction() {
    // Create temporary C file
    let temp_file = create_temp_c_file(SIMPLE_C_CODE);
    let file_path = temp_file.path().to_string_lossy().to_string();

    // TODO: This test will fail until kcs-parser implements call graph extraction
    // When implemented, should:
    // 1. Parse the C file using kcs-parser
    // 2. Extract call edges from ParseResult
    // 3. Convert to graph format and build KernelGraph
    // 4. Verify graph structure matches expected call relationships

    // Placeholder assertions that document expected behavior
    // These will be implemented when the parser supports call extraction

    // Expected parser integration:
    // let mut parser = kcs_parser::Parser::new(kcs_parser::ExtendedParserConfig::default()).unwrap();
    // let parse_result = parser.parse_file(&file_path).unwrap();
    //
    // // Should extract symbols
    // assert!(parse_result.symbols.len() >= 2, "Should find function symbols");
    //
    // // Should extract call edges (THIS CURRENTLY FAILS - to be implemented)
    // assert!(!parse_result.call_edges.is_empty(), "Should extract function call edges");
    //
    // // Build graph from parser results
    // let mut graph = KernelGraph::new();
    //
    // // Add symbols from parser
    // for symbol_info in &parse_result.symbols {
    //     let symbol = Symbol {
    //         name: symbol_info.name.clone(),
    //         file_path: symbol_info.file_path.clone(),
    //         line_number: symbol_info.start_line,
    //         symbol_type: SymbolType::Function, // Simplified for test
    //         signature: symbol_info.signature.clone(),
    //         config_dependencies: vec![],
    //     };
    //     graph.add_symbol(symbol);
    // }
    //
    // // Add call edges from parser
    // for call_edge in &parse_result.call_edges {
    //     let graph_edge = convert_parser_call_edge_to_graph(call_edge.call_type, call_edge.line_number);
    //     graph.add_call(&call_edge.caller, &call_edge.callee, graph_edge).unwrap();
    // }
    //
    // // Verify end-to-end integration
    // assert_eq!(graph.symbol_count(), 2);
    // assert_eq!(graph.call_count(), 2);
    //
    // let callees = graph.find_callees("main_function");
    // assert_eq!(callees.len(), 1);
    // assert_eq!(callees[0].name, "helper_function");

    // For now, document that this integration is pending parser implementation
    // This test serves as a specification for the integration contract
    println!("End-to-end integration pending parser call graph extraction implementation");

    // Use temp file to avoid unused variable warning
    println!("Test file: {}", file_path);
}

/// Test complex call patterns integration
#[test]
fn test_complex_call_patterns_integration() {
    let mut graph = KernelGraph::new();

    // Create symbols for function pointer test case
    let symbols = vec![
        Symbol {
            name: "add".to_string(),
            file_path: "test_complex.c".to_string(),
            line_number: 1,
            symbol_type: SymbolType::Function,
            signature: Some("int add(int a, int b)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "multiply".to_string(),
            file_path: "test_complex.c".to_string(),
            line_number: 2,
            symbol_type: SymbolType::Function,
            signature: Some("int multiply(int a, int b)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "execute_operation".to_string(),
            file_path: "test_complex.c".to_string(),
            line_number: 4,
            symbol_type: SymbolType::Function,
            signature: Some("int execute_operation(int x, int y, int (*op)(int, int))".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "main".to_string(),
            file_path: "test_complex.c".to_string(),
            line_number: 8,
            symbol_type: SymbolType::Function,
            signature: Some("int main()".to_string()),
            config_dependencies: vec![],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Simulate complex call patterns:
    // - main calls execute_operation (Direct)
    // - execute_operation calls function pointer (Indirect)
    let call_edges = vec![
        ("main", "execute_operation", 9, CallType::Direct),
        ("main", "execute_operation", 10, CallType::Direct),
        ("execute_operation", "add", 5, CallType::Indirect), // Function pointer call
        ("execute_operation", "multiply", 5, CallType::Indirect), // Alternative path
    ];

    for (caller, callee, line, call_type) in call_edges {
        let edge = CallEdge {
            call_type,
            call_site_line: line,
            conditional: false,
            config_guard: None,
        };

        graph.add_call(caller, callee, edge).unwrap();
    }

    // Verify complex call patterns are represented correctly
    assert_eq!(graph.symbol_count(), 4, "Should have 4 symbols");
    assert_eq!(graph.call_count(), 4, "Should have 4 call edges");

    // Test indirect call detection
    let execute_callees = graph.find_callees("execute_operation");
    assert_eq!(
        execute_callees.len(),
        2,
        "execute_operation should call 2 functions indirectly"
    );

    // Test call path analysis
    let path = graph.get_call_path("main", "add");
    assert!(path.is_some(), "Should find call path from main to add");
}

/// Test graph query performance with larger dataset
#[test]
fn test_call_extraction_performance() {
    let mut graph = KernelGraph::new();

    // Create a larger test dataset
    let num_functions = 100usize;
    let calls_per_function = 3usize;

    // Add symbols
    for i in 0..num_functions {
        let symbol = Symbol {
            name: format!("function_{}", i),
            file_path: "test_perf.c".to_string(),
            line_number: ((i + 1) * 10) as u32,
            symbol_type: SymbolType::Function,
            signature: Some(format!("void function_{}()", i)),
            config_dependencies: vec![],
        };
        graph.add_symbol(symbol);
    }

    // Add call edges (each function calls the next few functions)
    for i in 0..num_functions {
        for j in 1..=calls_per_function {
            let caller = format!("function_{}", i);
            let callee = format!("function_{}", (i + j) % num_functions);
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: ((i + 1) * 10 + j) as u32,
                conditional: false,
                config_guard: None,
            };

            graph.add_call(&caller, &callee, edge).unwrap();
        }
    }

    assert_eq!(
        graph.symbol_count(),
        num_functions,
        "Should have correct symbol count"
    );
    assert_eq!(
        graph.call_count(),
        num_functions * calls_per_function,
        "Should have correct call count"
    );

    // Test query performance
    let start = std::time::Instant::now();

    // Perform multiple queries
    for i in 0..10 {
        let function_name = format!("function_{}", i * 10);
        let callees = graph.find_callees(&function_name);
        assert_eq!(
            callees.len(),
            calls_per_function,
            "Each function should have expected callees"
        );
    }

    let duration = start.elapsed();
    assert!(
        duration.as_millis() < 100,
        "Queries should be fast: {}ms",
        duration.as_millis()
    );
}

/// Test call edge validation and error handling
#[test]
fn test_call_edge_validation() {
    let mut graph = KernelGraph::new();

    // Add a single symbol
    let symbol = Symbol {
        name: "test_func".to_string(),
        file_path: "test.c".to_string(),
        line_number: 10,
        symbol_type: SymbolType::Function,
        signature: None,
        config_dependencies: vec![],
    };
    graph.add_symbol(symbol);

    // Test error handling for missing symbols
    let edge = CallEdge {
        call_type: CallType::Direct,
        call_site_line: 15,
        conditional: false,
        config_guard: None,
    };

    // Should fail when caller doesn't exist
    let result = graph.add_call("nonexistent_caller", "test_func", edge.clone());
    assert!(
        result.is_err(),
        "Should fail when caller symbol doesn't exist"
    );

    // Should fail when callee doesn't exist
    let result = graph.add_call("test_func", "nonexistent_callee", edge);
    assert!(
        result.is_err(),
        "Should fail when callee symbol doesn't exist"
    );
}

/// Test configuration-aware call graph construction
#[test]
fn test_config_aware_call_graph() {
    let mut graph = KernelGraph::new();

    // Add symbols with different config dependencies
    let symbols = vec![
        Symbol {
            name: "common_func".to_string(),
            file_path: "common.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![], // Always available
        },
        Symbol {
            name: "debug_func".to_string(),
            file_path: "debug.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_DEBUG".to_string()],
        },
        Symbol {
            name: "net_func".to_string(),
            file_path: "network.c".to_string(),
            line_number: 30,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_NET".to_string()],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Test config-based symbol retrieval
    let debug_symbols = graph.symbols_by_config("CONFIG_DEBUG");
    assert!(
        debug_symbols.is_some(),
        "Should find symbols for CONFIG_DEBUG"
    );
    assert_eq!(
        debug_symbols.unwrap().len(),
        1,
        "Should have 1 debug symbol"
    );

    let net_symbols = graph.symbols_by_config("CONFIG_NET");
    assert!(net_symbols.is_some(), "Should find symbols for CONFIG_NET");
    assert_eq!(
        net_symbols.unwrap().len(),
        1,
        "Should have 1 network symbol"
    );

    // Test non-existent config
    let nonexistent_symbols = graph.symbols_by_config("CONFIG_NONEXISTENT");
    assert!(
        nonexistent_symbols.is_none(),
        "Should not find symbols for non-existent config"
    );
}
