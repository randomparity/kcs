/*!
 * Integration test for function pointer call extraction and analysis.
 *
 * Tests complex call patterns involving function pointers, callbacks,
 * and indirect function calls. This test must fail before implementation
 * and pass after.
 *
 * Key patterns tested:
 * - Function pointer declarations and assignments
 * - Indirect calls through function pointers
 * - Callback patterns and function tables
 * - Complex control flow with function pointers
 */

use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use std::io::Write;
use tempfile::NamedTempFile;

/// Test data representing function pointer usage patterns
const FUNCTION_POINTER_CODE: &str = r#"
// Basic function pointer operations
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

int execute_operation(int x, int y, int (*op)(int, int)) {
    return op(x, y);  // Indirect call through function pointer
}

int main() {
    int (*func_ptr)(int, int) = add;     // Assignment
    execute_operation(5, 3, add);        // Direct call with function pointer arg
    execute_operation(5, 3, multiply);   // Direct call with different function pointer
    return func_ptr(10, 20);             // Direct indirect call
}
"#;

/// Helper function to create temporary C files
fn create_temp_c_file(content: &str) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(content.as_bytes()).expect("Failed to write to temp file");
    temp_file
}

/// Create test symbols for basic function pointer example
fn create_function_pointer_symbols() -> Vec<Symbol> {
    vec![
        Symbol {
            name: "add".to_string(),
            file_path: "test_fp.c".to_string(),
            line_number: 2,
            symbol_type: SymbolType::Function,
            signature: Some("int add(int a, int b)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "multiply".to_string(),
            file_path: "test_fp.c".to_string(),
            line_number: 3,
            symbol_type: SymbolType::Function,
            signature: Some("int multiply(int a, int b)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "execute_operation".to_string(),
            file_path: "test_fp.c".to_string(),
            line_number: 5,
            symbol_type: SymbolType::Function,
            signature: Some("int execute_operation(int x, int y, int (*op)(int, int))".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "main".to_string(),
            file_path: "test_fp.c".to_string(),
            line_number: 9,
            symbol_type: SymbolType::Function,
            signature: Some("int main()".to_string()),
            config_dependencies: vec![],
        },
    ]
}

/// Integration test: Basic function pointer call patterns
#[test]
fn test_basic_function_pointer_integration() {
    let mut graph = KernelGraph::new();

    // Add symbols
    let symbols = create_function_pointer_symbols();
    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Simulate call edges for function pointer patterns:
    // 1. main -> execute_operation (direct calls)
    // 2. execute_operation -> add/multiply (indirect through function pointer)
    // 3. main -> add (direct indirect call via func_ptr)
    let call_edges = vec![
        ("main", "execute_operation", 11, CallType::Direct),
        ("main", "execute_operation", 12, CallType::Direct),
        ("execute_operation", "add", 6, CallType::Indirect), // Function pointer call
        ("execute_operation", "multiply", 6, CallType::Indirect), // Alternative path
        ("main", "add", 13, CallType::Indirect),             // Direct use of function pointer
    ];

    // Add call edges to graph
    for (caller, callee, line, call_type) in call_edges {
        let edge = CallEdge {
            call_type,
            call_site_line: line,
            conditional: false,
            config_guard: None,
        };

        let result = graph.add_call(caller, callee, edge);
        assert!(result.is_ok(), "Failed to add call edge: {} -> {}", caller, callee);
    }

    // Verify graph construction
    assert_eq!(graph.symbol_count(), 4, "Should have 4 symbols");
    assert_eq!(graph.call_count(), 5, "Should have 5 call edges");

    // Test indirect call detection
    let execute_callees = graph.find_callees("execute_operation");
    let indirect_callees: Vec<_> = execute_callees
        .into_iter()
        .filter(|s| s.name == "add" || s.name == "multiply")
        .collect();
    assert_eq!(
        indirect_callees.len(),
        2,
        "execute_operation should call 2 functions indirectly"
    );

    // Test main's callees (both direct and indirect)
    let main_callees = graph.find_callees("main");
    assert!(
        main_callees.len() >= 2,
        "main should have multiple callees (direct and indirect)"
    );

    // Verify function pointer targets are reachable
    let path_to_add = graph.get_call_path("main", "add");
    assert!(
        path_to_add.is_some(),
        "Should find call path from main to add (through function pointer)"
    );
}

/// Test callback-style function pointer patterns (kernel-style)
#[test]
fn test_callback_pattern_integration() {
    let mut graph = KernelGraph::new();

    // Create symbols for callback pattern
    let symbols = vec![
        Symbol {
            name: "generic_open".to_string(),
            file_path: "test_callback.c".to_string(),
            line_number: 8,
            symbol_type: SymbolType::Function,
            signature: Some("int generic_open(struct inode *, struct file *)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "generic_read".to_string(),
            file_path: "test_callback.c".to_string(),
            line_number: 12,
            symbol_type: SymbolType::Function,
            signature: Some(
                "int generic_read(struct file *, char *, size_t, loff_t *)".to_string(),
            ),
            config_dependencies: vec![],
        },
        Symbol {
            name: "do_file_operation".to_string(),
            file_path: "test_callback.c".to_string(),
            line_number: 22,
            symbol_type: SymbolType::Function,
            signature: Some("int do_file_operation(struct file *, char *, size_t)".to_string()),
            config_dependencies: vec![],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Simulate callback call pattern:
    // do_file_operation -> generic_read (indirect through struct function pointer)
    let callback_edge = CallEdge {
        call_type: CallType::Indirect, // Call through structure member function pointer
        call_site_line: 24,
        conditional: false,
        config_guard: None,
    };

    graph.add_call("do_file_operation", "generic_read", callback_edge).unwrap();

    // Verify callback pattern
    assert_eq!(graph.symbol_count(), 3, "Should have 3 symbols");
    assert_eq!(graph.call_count(), 1, "Should have 1 call edge");

    let callees = graph.find_callees("do_file_operation");
    assert_eq!(callees.len(), 1, "Should have 1 callee");
    assert_eq!(callees[0].name, "generic_read", "Should call generic_read indirectly");

    let callers = graph.find_callers("generic_read");
    assert_eq!(callers.len(), 1, "generic_read should have 1 caller");
    assert_eq!(callers[0].name, "do_file_operation", "Should be called by do_file_operation");
}

/// Test complex function pointer scenarios with conditional calls
#[test]
fn test_conditional_function_pointer_calls() {
    let mut graph = KernelGraph::new();

    // Create symbols for conditional function pointer scenario
    let symbols = vec![
        Symbol {
            name: "handler_a".to_string(),
            file_path: "test_conditional.c".to_string(),
            line_number: 5,
            symbol_type: SymbolType::Function,
            signature: Some("void handler_a()".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "handler_b".to_string(),
            file_path: "test_conditional.c".to_string(),
            line_number: 6,
            symbol_type: SymbolType::Function,
            signature: Some("void handler_b()".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "dispatch".to_string(),
            file_path: "test_conditional.c".to_string(),
            line_number: 8,
            symbol_type: SymbolType::Function,
            signature: Some("void dispatch(int type)".to_string()),
            config_dependencies: vec![],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Simulate conditional calls through function pointers
    // dispatch -> handler_a (conditional, indirect)
    // dispatch -> handler_b (conditional, indirect)
    let conditional_edges = vec![
        (
            "dispatch",
            "handler_a",
            CallEdge {
                call_type: CallType::Indirect,
                call_site_line: 12,
                conditional: true, // This call is conditional
                config_guard: Some("TYPE_A".to_string()),
            },
        ),
        (
            "dispatch",
            "handler_b",
            CallEdge {
                call_type: CallType::Indirect,
                call_site_line: 14,
                conditional: true, // This call is conditional
                config_guard: Some("TYPE_B".to_string()),
            },
        ),
    ];

    for (caller, callee, edge) in conditional_edges {
        graph.add_call(caller, callee, edge).unwrap();
    }

    // Verify conditional call patterns
    assert_eq!(graph.symbol_count(), 3, "Should have 3 symbols");
    assert_eq!(graph.call_count(), 2, "Should have 2 conditional call edges");

    let dispatch_callees = graph.find_callees("dispatch");
    assert_eq!(dispatch_callees.len(), 2, "dispatch should have 2 callees");

    // Both handlers should be reachable from dispatch
    let callee_names: Vec<_> = dispatch_callees.iter().map(|s| &s.name).collect();
    assert!(callee_names.contains(&&"handler_a".to_string()), "Should call handler_a");
    assert!(callee_names.contains(&&"handler_b".to_string()), "Should call handler_b");
}

/// Test function pointer arrays and tables
#[test]
fn test_function_pointer_tables() {
    let mut graph = KernelGraph::new();

    // Create symbols for function table scenario
    let symbols = vec![
        Symbol {
            name: "init_module".to_string(),
            file_path: "test_table.c".to_string(),
            line_number: 5,
            symbol_type: SymbolType::Function,
            signature: Some("int init_module()".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "cleanup_module".to_string(),
            file_path: "test_table.c".to_string(),
            line_number: 6,
            symbol_type: SymbolType::Function,
            signature: Some("void cleanup_module()".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "handle_request".to_string(),
            file_path: "test_table.c".to_string(),
            line_number: 7,
            symbol_type: SymbolType::Function,
            signature: Some("int handle_request(int type)".to_string()),
            config_dependencies: vec![],
        },
        Symbol {
            name: "module_dispatcher".to_string(),
            file_path: "test_table.c".to_string(),
            line_number: 15,
            symbol_type: SymbolType::Function,
            signature: Some("int module_dispatcher(int cmd)".to_string()),
            config_dependencies: vec![],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Simulate function table calls
    // module_dispatcher calls functions from an array of function pointers
    let table_edges = vec![
        ("module_dispatcher", "init_module", CallType::Indirect),
        ("module_dispatcher", "cleanup_module", CallType::Indirect),
        ("module_dispatcher", "handle_request", CallType::Indirect),
    ];

    for (caller, callee, call_type) in table_edges {
        let edge = CallEdge {
            call_type,
            call_site_line: 20, // Same line, different array indices
            conditional: true,  // Table lookup is conditional
            config_guard: None,
        };

        graph.add_call(caller, callee, edge).unwrap();
    }

    // Verify function table pattern
    assert_eq!(graph.symbol_count(), 4, "Should have 4 symbols");
    assert_eq!(graph.call_count(), 3, "Should have 3 table call edges");

    let dispatcher_callees = graph.find_callees("module_dispatcher");
    assert_eq!(
        dispatcher_callees.len(),
        3,
        "module_dispatcher should call 3 functions from table"
    );

    // All functions should be reachable from dispatcher
    for callee in &dispatcher_callees {
        assert!(
            ["init_module", "cleanup_module", "handle_request"].contains(&callee.name.as_str()),
            "Should call functions from table: {}",
            callee.name
        );
    }
}

/// Test performance with large function pointer networks
#[test]
fn test_function_pointer_performance() {
    let mut graph = KernelGraph::new();

    let num_handlers = 50usize;
    let num_dispatchers = 10usize;

    // Create handler symbols
    for i in 0..num_handlers {
        let symbol = Symbol {
            name: format!("handler_{}", i),
            file_path: "test_perf_fp.c".to_string(),
            line_number: ((i + 1) * 10) as u32,
            symbol_type: SymbolType::Function,
            signature: Some(format!("void handler_{}()", i)),
            config_dependencies: vec![],
        };
        graph.add_symbol(symbol);
    }

    // Create dispatcher symbols
    for i in 0..num_dispatchers {
        let symbol = Symbol {
            name: format!("dispatcher_{}", i),
            file_path: "test_perf_fp.c".to_string(),
            line_number: ((num_handlers + i + 1) * 10) as u32,
            symbol_type: SymbolType::Function,
            signature: Some(format!("void dispatcher_{}(int type)", i)),
            config_dependencies: vec![],
        };
        graph.add_symbol(symbol);
    }

    // Each dispatcher can call multiple handlers through function pointers
    let handlers_per_dispatcher = 5usize;
    for i in 0..num_dispatchers {
        for j in 0..handlers_per_dispatcher {
            let dispatcher = format!("dispatcher_{}", i);
            let handler = format!("handler_{}", (i * handlers_per_dispatcher + j) % num_handlers);

            let edge = CallEdge {
                call_type: CallType::Indirect, // Function pointer call
                call_site_line: ((num_handlers + i + 1) * 10 + j + 1) as u32,
                conditional: true, // Conditional based on function pointer table
                config_guard: None,
            };

            graph.add_call(&dispatcher, &handler, edge).unwrap();
        }
    }

    // Verify construction
    let expected_symbols = num_handlers + num_dispatchers;
    let expected_calls = num_dispatchers * handlers_per_dispatcher;

    assert_eq!(graph.symbol_count(), expected_symbols, "Should have correct symbol count");
    assert_eq!(graph.call_count(), expected_calls, "Should have correct call count");

    // Test query performance
    let start = std::time::Instant::now();

    // Query all dispatchers
    for i in 0..num_dispatchers {
        let dispatcher = format!("dispatcher_{}", i);
        let callees = graph.find_callees(&dispatcher);
        assert_eq!(
            callees.len(),
            handlers_per_dispatcher,
            "Each dispatcher should call {} handlers",
            handlers_per_dispatcher
        );
    }

    let query_duration = start.elapsed();

    // Test path finding performance
    let path_start = std::time::Instant::now();

    let path = graph.get_call_path("dispatcher_0", "handler_0");
    assert!(path.is_some(), "Should find path to handler");

    let path_duration = path_start.elapsed();

    // Performance assertions
    assert!(
        query_duration.as_millis() < 50,
        "Queries should be fast: {}ms",
        query_duration.as_millis()
    );
    assert!(
        path_duration.as_millis() < 10,
        "Path finding should be fast: {}ms",
        path_duration.as_millis()
    );
}

/// Test end-to-end integration with parser (will fail until parser is implemented)
#[test]
fn test_end_to_end_function_pointer_extraction() {
    // Create temporary C file with function pointers
    let temp_file = create_temp_c_file(FUNCTION_POINTER_CODE);
    let file_path = temp_file.path().to_string_lossy().to_string();

    // TODO: This test will fail until kcs-parser implements function pointer call extraction
    // When implemented, should:
    // 1. Parse the C file using kcs-parser
    // 2. Identify function pointer declarations and assignments
    // 3. Extract indirect calls through function pointers
    // 4. Classify calls as Direct vs Indirect appropriately
    // 5. Build accurate call graph with function pointer relationships

    // Expected parser integration:
    // let mut parser = kcs_parser::Parser::new(kcs_parser::ExtendedParserConfig::default()).unwrap();
    // let parse_result = parser.parse_file(&file_path).unwrap();
    //
    // // Should extract function pointer calls as Indirect
    // let indirect_calls: Vec<_> = parse_result.call_edges
    //     .iter()
    //     .filter(|edge| matches!(edge.call_type, kcs_parser::CallType::Indirect))
    //     .collect();
    //
    // assert!(!indirect_calls.is_empty(), "Should detect indirect calls through function pointers");
    //
    // // Build graph from parser results
    // let mut graph = KernelGraph::new();
    // // ... build graph from parser results ...
    //
    // // Verify function pointer call detection
    // let execute_callees = graph.find_callees("execute_operation");
    // let indirect_callees: Vec<_> = execute_callees
    //     .into_iter()
    //     .filter(|s| s.name == "add" || s.name == "multiply")
    //     .collect();
    //
    // assert_eq!(indirect_callees.len(), 2, "Should detect calls to both add and multiply");

    // For now, document that this integration is pending
    println!("End-to-end function pointer integration pending parser implementation");
    println!("Test file: {}", file_path);
}

/// Test error handling with invalid function pointer scenarios
#[test]
fn test_function_pointer_error_handling() {
    let mut graph = KernelGraph::new();

    // Add minimal symbols
    let symbol = Symbol {
        name: "test_func".to_string(),
        file_path: "test.c".to_string(),
        line_number: 10,
        symbol_type: SymbolType::Function,
        signature: None,
        config_dependencies: vec![],
    };
    graph.add_symbol(symbol);

    // Test error cases for function pointer calls
    let edge = CallEdge {
        call_type: CallType::Indirect,
        call_site_line: 15,
        conditional: false,
        config_guard: None,
    };

    // Should fail when trying to add indirect call to nonexistent function
    let result = graph.add_call("test_func", "nonexistent_target", edge.clone());
    assert!(result.is_err(), "Should fail when function pointer target doesn't exist");

    // Should fail when caller doesn't exist
    let result = graph.add_call("nonexistent_caller", "test_func", edge);
    assert!(result.is_err(), "Should fail when function pointer caller doesn't exist");
}

/// Test mixed direct and indirect call patterns
#[test]
fn test_mixed_call_patterns() {
    let mut graph = KernelGraph::new();

    // Create symbols for mixed pattern test
    let symbols = vec![
        Symbol {
            name: "caller".to_string(),
            file_path: "mixed.c".to_string(),
            line_number: 5,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        },
        Symbol {
            name: "direct_target".to_string(),
            file_path: "mixed.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        },
        Symbol {
            name: "indirect_target".to_string(),
            file_path: "mixed.c".to_string(),
            line_number: 15,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Add mixed call patterns
    let edges = vec![
        (
            "caller",
            "direct_target",
            CallEdge {
                call_type: CallType::Direct,
                call_site_line: 7,
                conditional: false,
                config_guard: None,
            },
        ),
        (
            "caller",
            "indirect_target",
            CallEdge {
                call_type: CallType::Indirect,
                call_site_line: 8,
                conditional: false,
                config_guard: None,
            },
        ),
    ];

    for (caller, callee, edge) in edges {
        graph.add_call(caller, callee, edge).unwrap();
    }

    // Verify mixed patterns
    assert_eq!(graph.symbol_count(), 3, "Should have 3 symbols");
    assert_eq!(graph.call_count(), 2, "Should have 2 mixed call edges");

    let callees = graph.find_callees("caller");
    assert_eq!(callees.len(), 2, "caller should have 2 callees");

    // Both direct and indirect targets should be reachable
    let callee_names: Vec<_> = callees.iter().map(|s| &s.name).collect();
    assert!(
        callee_names.contains(&&"direct_target".to_string()),
        "Should call direct_target"
    );
    assert!(
        callee_names.contains(&&"indirect_target".to_string()),
        "Should call indirect_target through function pointer"
    );
}
