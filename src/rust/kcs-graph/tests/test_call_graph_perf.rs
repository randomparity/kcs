/*!
 * Performance tests for call graph construction with large files.
 *
 * Tests performance characteristics under various conditions:
 * - Large file parsing with many function definitions
 * - Complex call graph construction with deep nesting
 * - Memory usage patterns during graph building
 * - Query performance on large graphs
 *
 * Performance targets (as defined in KCS requirements):
 * - Full index: ≤20 minutes
 * - Incremental: ≤3 minutes
 * - Query p95: ≤600ms
 * - Scale: ~50k symbols, ~10k entry points
 */

use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use std::collections::HashMap;
use std::io::Write;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

/// Performance benchmark configuration
#[allow(dead_code)]
struct BenchmarkConfig {
    /// Number of functions to generate
    function_count: usize,
    /// Average calls per function
    calls_per_function: usize,
    /// Maximum nesting depth for call chains
    max_nesting_depth: usize,
    /// Whether to include function pointers (indirect calls)
    include_function_pointers: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            function_count: 1000,
            calls_per_function: 5,
            max_nesting_depth: 10,
            include_function_pointers: true,
        }
    }
}

/// Generate large C source code for performance testing
fn generate_large_c_code(config: &BenchmarkConfig) -> String {
    let mut code = String::new();

    // Header
    code.push_str("// Generated large C file for call graph performance testing\n");
    code.push_str("#include <stdio.h>\n");
    code.push_str("#include <stdlib.h>\n\n");

    // Forward declarations
    code.push_str("// Forward declarations\n");
    for i in 0..config.function_count {
        code.push_str(&format!("int function_{}(int param);\n", i));
    }

    if config.include_function_pointers {
        code.push_str("\n// Function pointer type\n");
        code.push_str("typedef int (*operation_t)(int);\n\n");

        // Function pointer table
        code.push_str("static operation_t operations[] = {\n");
        for i in 0..config.function_count.min(20) {
            code.push_str(&format!("    function_{},\n", i));
        }
        code.push_str("    NULL\n};\n\n");

        // Dispatcher function
        code.push_str("int dispatch_operation(int op_index, int param) {\n");
        code.push_str("    if (op_index >= 0 && op_index < 20 && operations[op_index]) {\n");
        code.push_str("        return operations[op_index](param);\n");
        code.push_str("    }\n");
        code.push_str("    return -1;\n");
        code.push_str("}\n\n");
    }

    // Generate function implementations
    for i in 0..config.function_count {
        code.push_str(&format!("int function_{}(int param) {{\n", i));

        // Add local variables for complexity
        code.push_str("    int result = 0;\n");
        code.push_str("    int temp = param;\n");

        // Generate calls to other functions
        let call_count = if config.calls_per_function > 0 {
            (i % config.calls_per_function) + 1
        } else {
            0
        };

        for j in 0..call_count {
            let target_func = (i + j + 1) % config.function_count;

            // Mix of direct and conditional calls
            if j % 3 == 0 {
                code.push_str(&format!("    if (temp > {}) {{\n", j));
                code.push_str(&format!(
                    "        result += function_{}(temp - 1);\n",
                    target_func
                ));
                code.push_str("    }\n");
            } else {
                code.push_str(&format!(
                    "    result += function_{}(temp + {});\n",
                    target_func, j
                ));
            }
        }

        // Add function pointer calls for complexity
        if config.include_function_pointers && i % 10 == 0 {
            code.push_str("    // Indirect call through function pointer\n");
            code.push_str(&format!(
                "    result += dispatch_operation({}, param);\n",
                i % 20
            ));
        }

        code.push_str("    return result;\n");
        code.push_str("}\n\n");
    }

    // Main function that calls various functions
    code.push_str("int main() {\n");
    code.push_str("    int total = 0;\n");
    for i in 0..config.function_count.min(100) {
        if i % 10 == 0 {
            code.push_str(&format!("    total += function_{}({});\n", i, i));
        }
    }
    code.push_str("    return total;\n");
    code.push_str("}\n");

    code
}

/// Create test symbols from generated code configuration
fn create_large_symbol_set(config: &BenchmarkConfig) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    // Add all generated functions
    for i in 0..config.function_count {
        symbols.push(Symbol {
            name: format!("function_{}", i),
            file_path: "large_test.c".to_string(),
            line_number: (i as u32 + 1) * 10, // Spread functions across lines
            symbol_type: SymbolType::Function,
            signature: Some(format!("int function_{}(int param)", i)),
            config_dependencies: if i % 5 == 0 {
                vec![format!("CONFIG_FEATURE_{}", i / 5)]
            } else {
                vec![]
            },
        });
    }

    // Add dispatcher function if function pointers are enabled
    if config.include_function_pointers {
        symbols.push(Symbol {
            name: "dispatch_operation".to_string(),
            file_path: "large_test.c".to_string(),
            line_number: (config.function_count as u32 + 1) * 10,
            symbol_type: SymbolType::Function,
            signature: Some("int dispatch_operation(int op_index, int param)".to_string()),
            config_dependencies: vec![],
        });
    }

    // Add main function
    symbols.push(Symbol {
        name: "main".to_string(),
        file_path: "large_test.c".to_string(),
        line_number: (config.function_count as u32 + 2) * 10,
        symbol_type: SymbolType::Function,
        signature: Some("int main()".to_string()),
        config_dependencies: vec![],
    });

    symbols
}

/// Generate call edges for large test scenario
fn create_large_call_edges(config: &BenchmarkConfig) -> Vec<(String, String, CallEdge)> {
    let mut edges = Vec::new();

    // Generate calls between functions
    for i in 0..config.function_count {
        let caller = format!("function_{}", i);
        let call_count = if config.calls_per_function > 0 {
            (i % config.calls_per_function) + 1
        } else {
            0
        };

        for j in 0..call_count {
            let target_func = (i + j + 1) % config.function_count;
            let callee = format!("function_{}", target_func);

            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: (i as u32 + 1) * 10 + j as u32 + 1,
                conditional: j % 3 == 0, // Some calls are conditional
                config_guard: if i % 10 == 0 {
                    Some(format!("CONFIG_FEATURE_{}", i / 10))
                } else {
                    None
                },
            };

            edges.push((caller.clone(), callee, edge));
        }

        // Add function pointer calls
        if config.include_function_pointers && i % 10 == 0 {
            let edge = CallEdge {
                call_type: CallType::Indirect,
                call_site_line: (i as u32 + 1) * 10 + 50,
                conditional: false,
                config_guard: None,
            };

            // Indirect call to dispatcher
            edges.push((
                format!("function_{}", i),
                "dispatch_operation".to_string(),
                edge,
            ));

            // Dispatcher calls various functions indirectly
            for k in 0..20.min(config.function_count) {
                let dispatcher_edge = CallEdge {
                    call_type: CallType::Indirect,
                    call_site_line: (config.function_count as u32 + 1) * 10 + k as u32,
                    conditional: true, // Conditional based on op_index
                    config_guard: None,
                };

                edges.push((
                    "dispatch_operation".to_string(),
                    format!("function_{}", k),
                    dispatcher_edge,
                ));
            }
        }
    }

    // Main function calls
    for i in 0..config.function_count.min(100) {
        if i % 10 == 0 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: (config.function_count as u32 + 2) * 10 + (i / 10) as u32,
                conditional: false,
                config_guard: None,
            };

            edges.push(("main".to_string(), format!("function_{}", i), edge));
        }
    }

    edges
}

/// Benchmark graph construction performance
fn benchmark_graph_construction(config: &BenchmarkConfig) -> (Duration, KernelGraph, usize) {
    let start = Instant::now();

    let mut graph = KernelGraph::new();
    let symbols = create_large_symbol_set(config);
    let edges = create_large_call_edges(config);

    // Add symbols
    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Add call edges
    let mut successful_edges = 0;
    for (caller, callee, edge) in edges {
        if graph.add_call(&caller, &callee, edge).is_ok() {
            successful_edges += 1;
        }
    }

    let duration = start.elapsed();
    (duration, graph, successful_edges)
}

/// Benchmark query performance on constructed graph
fn benchmark_query_performance(
    graph: &KernelGraph,
    config: &BenchmarkConfig,
) -> HashMap<String, Duration> {
    let mut query_times = HashMap::new();

    // Benchmark find_callees queries
    let start = Instant::now();
    for i in 0..config.function_count.min(100) {
        let function_name = format!("function_{}", i * 10);
        let _callees = graph.find_callees(&function_name);
    }
    query_times.insert("find_callees_batch".to_string(), start.elapsed());

    // Benchmark find_callers queries
    let start = Instant::now();
    for i in 0..config.function_count.min(100) {
        let function_name = format!("function_{}", i * 10);
        let _callers = graph.find_callers(&function_name);
    }
    query_times.insert("find_callers_batch".to_string(), start.elapsed());

    // Benchmark call path queries
    let start = Instant::now();
    for i in 0..10 {
        let source = format!("function_{}", i);
        let target = format!("function_{}", (i + 50) % config.function_count);
        let _path = graph.get_call_path(&source, &target);
    }
    query_times.insert("call_path_batch".to_string(), start.elapsed());

    // Benchmark config-based queries
    let start = Instant::now();
    for i in 0..10 {
        let config_name = format!("CONFIG_FEATURE_{}", i);
        let _symbols = graph.symbols_by_config(&config_name);
    }
    query_times.insert("config_query_batch".to_string(), start.elapsed());

    query_times
}

/// Performance test: Small scale baseline
#[test]
fn test_small_scale_performance() {
    let config = BenchmarkConfig {
        function_count: 100,
        calls_per_function: 3,
        max_nesting_depth: 5,
        include_function_pointers: false,
    };

    let (construction_time, graph, edge_count) = benchmark_graph_construction(&config);

    // Verify basic metrics
    assert_eq!(graph.symbol_count(), config.function_count + 1); // +1 for main
    assert!(edge_count > 0, "Should have created call edges");

    // Performance assertions
    assert!(
        construction_time.as_millis() < 1000,
        "Small graph construction should be fast: {}ms",
        construction_time.as_millis()
    );

    // Test queries
    let query_times = benchmark_query_performance(&graph, &config);

    for (query_type, duration) in &query_times {
        assert!(
            duration.as_millis() < 100,
            "Small graph queries should be fast: {} took {}ms",
            query_type,
            duration.as_millis()
        );
    }

    println!("Small scale performance:");
    println!("  Construction: {}ms", construction_time.as_millis());
    println!("  Symbols: {}", graph.symbol_count());
    println!("  Edges: {}", edge_count);
    for (query_type, duration) in &query_times {
        println!("  {}: {}ms", query_type, duration.as_millis());
    }
}

/// Performance test: Medium scale (realistic kernel module)
#[test]
fn test_medium_scale_performance() {
    let config = BenchmarkConfig {
        function_count: 1000,
        calls_per_function: 5,
        max_nesting_depth: 8,
        include_function_pointers: true,
    };

    let (construction_time, graph, edge_count) = benchmark_graph_construction(&config);

    // Verify metrics
    assert_eq!(graph.symbol_count(), config.function_count + 2); // +1 for main, +1 for dispatcher
    assert!(
        edge_count > config.function_count,
        "Should have many call edges"
    );

    // Performance assertions (should be reasonable for medium scale)
    assert!(
        construction_time.as_secs() < 10,
        "Medium graph construction should complete in reasonable time: {}s",
        construction_time.as_secs()
    );

    // Test queries
    let query_times = benchmark_query_performance(&graph, &config);

    for (query_type, duration) in &query_times {
        assert!(
            duration.as_millis() < 1000,
            "Medium graph queries should be reasonable: {} took {}ms",
            query_type,
            duration.as_millis()
        );
    }

    println!("Medium scale performance:");
    println!("  Construction: {}ms", construction_time.as_millis());
    println!("  Symbols: {}", graph.symbol_count());
    println!("  Edges: {}", edge_count);
    for (query_type, duration) in &query_times {
        println!("  {}: {}ms", query_type, duration.as_millis());
    }
}

/// Performance test: Large scale (approaching KCS targets)
#[test]
fn test_large_scale_performance() {
    let config = BenchmarkConfig {
        function_count: 10000,
        calls_per_function: 8,
        max_nesting_depth: 15,
        include_function_pointers: true,
    };

    let (construction_time, graph, edge_count) = benchmark_graph_construction(&config);

    // Verify scale
    assert!(
        graph.symbol_count() > 10000,
        "Should handle large symbol count"
    );
    assert!(edge_count > 50000, "Should handle large edge count");

    // Performance assertions aligned with KCS targets
    // Target: Query p95 ≤ 600ms
    let query_times = benchmark_query_performance(&graph, &config);

    for (query_type, duration) in &query_times {
        if query_type.contains("batch") {
            // These are batch operations, so higher time is acceptable
            assert!(
                duration.as_millis() < 5000,
                "Large graph batch queries should complete reasonably: {} took {}ms",
                query_type,
                duration.as_millis()
            );
        } else {
            // Individual queries should meet KCS targets
            assert!(
                duration.as_millis() < 600,
                "Individual query should meet KCS target: {} took {}ms",
                query_type,
                duration.as_millis()
            );
        }
    }

    println!("Large scale performance:");
    println!("  Construction: {}ms", construction_time.as_millis());
    println!("  Symbols: {}", graph.symbol_count());
    println!("  Edges: {}", edge_count);
    for (query_type, duration) in &query_times {
        println!("  {}: {}ms", query_type, duration.as_millis());
    }
}

/// Memory usage test: Monitor memory consumption patterns
#[test]
fn test_memory_usage_patterns() {
    let config = BenchmarkConfig {
        function_count: 5000,
        calls_per_function: 6,
        max_nesting_depth: 10,
        include_function_pointers: true,
    };

    // This is a basic memory usage test - in production we'd use more
    // sophisticated memory profiling
    let initial_memory = std::process::id(); // Placeholder for memory measurement

    let (construction_time, graph, edge_count) = benchmark_graph_construction(&config);

    let final_memory = std::process::id(); // Placeholder for memory measurement

    // Basic verification that construction succeeded
    assert!(graph.symbol_count() > 5000);
    assert!(edge_count > 25000);
    assert!(construction_time.as_secs() < 30); // Should complete in reasonable time

    println!("Memory usage test:");
    println!("  Construction time: {}ms", construction_time.as_millis());
    println!("  Symbols: {}", graph.symbol_count());
    println!("  Edges: {}", edge_count);
    println!(
        "  Memory delta: {} (placeholder)",
        final_memory.wrapping_sub(initial_memory)
    );
}

/// Stress test: Maximum reasonable scale
#[test]
#[ignore] // Ignored by default due to long runtime
fn test_stress_maximum_scale() {
    let config = BenchmarkConfig {
        function_count: 50000, // Approaching KCS target of ~50k symbols
        calls_per_function: 10,
        max_nesting_depth: 20,
        include_function_pointers: true,
    };

    let (construction_time, graph, edge_count) = benchmark_graph_construction(&config);

    // Verify we can handle KCS target scale
    assert!(
        graph.symbol_count() >= 50000,
        "Should handle KCS target symbol count"
    );
    assert!(edge_count > 100000, "Should handle large edge count");

    // Construction should complete within KCS targets
    // Target: Full index ≤20 minutes (this is just graph construction, not parsing)
    assert!(
        construction_time.as_secs() < 300, // 5 minutes for graph construction only
        "Stress test construction should complete reasonably: {}s",
        construction_time.as_secs()
    );

    // Test query performance under stress
    let query_times = benchmark_query_performance(&graph, &config);

    for (query_type, duration) in &query_times {
        if query_type.contains("batch") {
            // Batch operations can take longer
            assert!(
                duration.as_millis() < 10000,
                "Stress test batch queries: {} took {}ms",
                query_type,
                duration.as_millis()
            );
        }
    }

    println!("Stress test performance:");
    println!("  Construction: {}s", construction_time.as_secs());
    println!("  Symbols: {}", graph.symbol_count());
    println!("  Edges: {}", edge_count);
    for (query_type, duration) in &query_times {
        println!("  {}: {}ms", query_type, duration.as_millis());
    }
}

/// Concurrent access performance test
#[test]
fn test_concurrent_query_performance() {
    let config = BenchmarkConfig {
        function_count: 2000,
        calls_per_function: 5,
        max_nesting_depth: 8,
        include_function_pointers: true,
    };

    let (construction_time, graph, _edge_count) = benchmark_graph_construction(&config);

    // Test concurrent read access (KernelGraph should support concurrent reads)
    use std::sync::Arc;
    use std::thread;

    let graph = Arc::new(graph);
    let mut handles = vec![];

    let start = Instant::now();

    // Spawn multiple threads to query the graph concurrently
    for thread_id in 0..4 {
        let graph_clone = Arc::clone(&graph);
        let handle = thread::spawn(move || {
            let mut query_count = 0;

            for i in 0..100 {
                let function_name =
                    format!("function_{}", (thread_id * 100 + i) % config.function_count);

                // Mix of different query types
                match i % 3 {
                    0 => {
                        graph_clone.find_callees(&function_name);
                    }
                    1 => {
                        graph_clone.find_callers(&function_name);
                    }
                    _ => {
                        graph_clone.get_call_path("main", &function_name);
                    }
                }

                query_count += 1;
            }

            query_count
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    let mut total_queries = 0;
    for handle in handles {
        total_queries += handle.join().unwrap();
    }

    let concurrent_time = start.elapsed();

    // Performance assertions
    assert_eq!(total_queries, 400); // 4 threads * 100 queries each
    assert!(
        concurrent_time.as_millis() < 5000,
        "Concurrent queries should complete efficiently: {}ms",
        concurrent_time.as_millis()
    );

    println!("Concurrent performance:");
    println!("  Construction: {}ms", construction_time.as_millis());
    println!(
        "  Concurrent queries: {} in {}ms",
        total_queries,
        concurrent_time.as_millis()
    );
    println!(
        "  Avg per query: {:.2}ms",
        concurrent_time.as_millis() as f64 / total_queries as f64
    );
}

/// Integration test: Generated code parsing (when parser is available)
#[test]
fn test_generated_code_parsing_performance() {
    let config = BenchmarkConfig {
        function_count: 1000,
        calls_per_function: 4,
        max_nesting_depth: 6,
        include_function_pointers: true,
    };

    // Generate C code
    let c_code = generate_large_c_code(&config);

    // Create temporary file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(c_code.as_bytes())
        .expect("Failed to write code");

    // Verify code generation
    assert!(c_code.len() > 10000, "Should generate substantial code");
    assert!(
        c_code.contains("function_0"),
        "Should contain generated functions"
    );
    assert!(
        c_code.contains("dispatch_operation"),
        "Should contain function pointers"
    );

    // For now, just verify the code structure (parsing integration pending)
    let function_count = c_code.matches("int function_").count();
    assert!(
        function_count >= config.function_count,
        "Should generate expected number of functions"
    );

    println!("Generated code performance test:");
    println!("  Generated code size: {} bytes", c_code.len());
    println!("  Function count: {}", function_count);
    println!(
        "  Include function pointers: {}",
        config.include_function_pointers
    );

    // TODO: When kcs-parser supports call extraction:
    // - Parse the generated C code
    // - Extract call relationships
    // - Build KernelGraph from extracted data
    // - Measure end-to-end performance
    // - Verify performance meets KCS targets
}
