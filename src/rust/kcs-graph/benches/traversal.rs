//! Performance benchmarks for graph traversal algorithms
//!
//! This module benchmarks various traversal operations on kernel call graphs
//! to ensure they meet the constitutional requirement of p95 ≤600ms for queries.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kcs_graph::{
    cycles::CycleDetector, traversal::GraphTraversal, CallEdge, CallType, KernelGraph, Symbol,
    SymbolType,
};
use std::time::Duration;

/// Create a small graph for basic benchmarks
fn create_small_graph() -> KernelGraph {
    let mut graph = KernelGraph::new();

    // Create 20 symbols
    for i in 0..20 {
        let symbol = Symbol {
            name: format!("func_{}", i),
            file_path: "test.c".to_string(),
            line_number: (i as u32 + 1) * 10,
            symbol_type: SymbolType::Function,
            signature: Some(format!("int func_{}(void)", i)),
            config_dependencies: vec![],
        };
        graph.add_symbol(symbol);
    }

    // Create a connected graph with some branches
    for i in 0..19 {
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 100 + i as u32,
            conditional: false,
            config_guard: None,
        };
        graph
            .add_call(&format!("func_{}", i), &format!("func_{}", i + 1), edge)
            .unwrap();

        // Add some cross-edges for complexity
        if i % 3 == 0 && i < 17 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 200 + i as u32,
                conditional: false,
                config_guard: None,
            };
            graph
                .add_call(&format!("func_{}", i), &format!("func_{}", i + 3), edge)
                .unwrap();
        }
    }

    graph
}

/// Create a medium-sized graph for more realistic benchmarks
fn create_medium_graph() -> KernelGraph {
    let mut graph = KernelGraph::new();

    // Create 500 symbols
    for i in 0..500 {
        let symbol = Symbol {
            name: format!("func_{}", i),
            file_path: format!("file_{}.c", i / 50),
            line_number: (i as u32 + 1) * 10,
            symbol_type: SymbolType::Function,
            signature: Some(format!("int func_{}(void)", i)),
            config_dependencies: if i % 5 == 0 {
                vec![format!("CONFIG_{}", i / 5)]
            } else {
                vec![]
            },
        };
        graph.add_symbol(symbol);
    }

    // Create a more complex graph structure
    for i in 0..499 {
        // Linear chain
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 1000 + i as u32,
            conditional: false,
            config_guard: None,
        };
        graph
            .add_call(&format!("func_{}", i), &format!("func_{}", i + 1), edge)
            .unwrap();

        // Add branches and cross-edges
        if i % 5 == 0 && i < 495 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 2000 + i as u32,
                conditional: true,
                config_guard: Some(format!("CONFIG_{}", i / 5)),
            };
            graph
                .add_call(&format!("func_{}", i), &format!("func_{}", i + 5), edge)
                .unwrap();
        }

        // Add some indirect calls
        if i % 7 == 0 && i < 493 {
            let edge = CallEdge {
                call_type: CallType::Indirect,
                call_site_line: 3000 + i as u32,
                conditional: false,
                config_guard: None,
            };
            graph
                .add_call(&format!("func_{}", i), &format!("func_{}", i + 7), edge)
                .unwrap();
        }

        // Add some back edges to create cycles
        if i % 20 == 19 && i >= 20 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 4000 + i as u32,
                conditional: true,
                config_guard: Some("CONFIG_RECURSIVE".to_string()),
            };
            graph
                .add_call(&format!("func_{}", i), &format!("func_{}", i - 20), edge)
                .unwrap();
        }
    }

    graph
}

/// Create a large graph to test scalability
fn create_large_graph() -> KernelGraph {
    let mut graph = KernelGraph::new();

    // Create 5000 symbols (simulating a real kernel subsystem)
    for i in 0..5000 {
        let symbol = Symbol {
            name: format!("kern_func_{}", i),
            file_path: format!("drivers/subsys_{}/file_{}.c", i / 500, i / 50),
            line_number: (i as u32 + 1) * 10,
            symbol_type: SymbolType::Function,
            signature: Some(format!("int kern_func_{}(void)", i)),
            config_dependencies: match i % 10 {
                0 => vec!["CONFIG_MODULE_A".to_string()],
                1 => vec!["CONFIG_MODULE_B".to_string()],
                2 => vec!["CONFIG_MODULE_A".to_string(), "CONFIG_MODULE_B".to_string()],
                _ => vec![],
            },
        };
        graph.add_symbol(symbol);
    }

    // Create a realistic call graph structure
    for i in 0..4999 {
        // Main call chain
        if i < 4999 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 10000 + i as u32,
                conditional: false,
                config_guard: None,
            };
            graph
                .add_call(&format!("kern_func_{}", i), &format!("kern_func_{}", i + 1), edge)
                .unwrap();
        }

        // Subsystem internal calls
        if i % 10 == 0 && i < 4990 {
            for j in 1..5 {
                let edge = CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 20000 + i as u32 + j,
                    conditional: false,
                    config_guard: None,
                };
                graph
                    .add_call(
                        &format!("kern_func_{}", i),
                        &format!("kern_func_{}", i + j as usize),
                        edge,
                    )
                    .unwrap();
            }
        }

        // Cross-subsystem calls
        if i % 100 == 0 && i < 4900 {
            let edge = CallEdge {
                call_type: CallType::FunctionPointer,
                call_site_line: 30000 + i as u32,
                conditional: false,
                config_guard: None,
            };
            graph
                .add_call(&format!("kern_func_{}", i), &format!("kern_func_{}", i + 100), edge)
                .unwrap();
        }
    }

    graph
}

fn benchmark_bfs_traversal(c: &mut Criterion) {
    let small_graph = create_small_graph();
    let medium_graph = create_medium_graph();
    let large_graph = create_large_graph();

    let mut group = c.benchmark_group("bfs_traversal");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("small_graph", |b| {
        let traversal = GraphTraversal::new(&small_graph);
        b.iter(|| {
            let result = traversal
                .bfs(black_box("func_0"), kcs_graph::traversal::TraversalOptions::default())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("medium_graph", |b| {
        let traversal = GraphTraversal::new(&medium_graph);
        b.iter(|| {
            let result = traversal
                .bfs(black_box("func_0"), kcs_graph::traversal::TraversalOptions::default())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("large_graph", |b| {
        let traversal = GraphTraversal::new(&large_graph);
        b.iter(|| {
            let result = traversal
                .bfs(black_box("kern_func_0"), kcs_graph::traversal::TraversalOptions::default())
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_dfs_traversal(c: &mut Criterion) {
    let small_graph = create_small_graph();
    let medium_graph = create_medium_graph();
    let large_graph = create_large_graph();

    let mut group = c.benchmark_group("dfs_traversal");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("small_graph", |b| {
        let traversal = GraphTraversal::new(&small_graph);
        b.iter(|| {
            let result = traversal
                .dfs(black_box("func_0"), kcs_graph::traversal::TraversalOptions::default())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("medium_graph", |b| {
        let traversal = GraphTraversal::new(&medium_graph);
        b.iter(|| {
            let result = traversal
                .dfs(black_box("func_0"), kcs_graph::traversal::TraversalOptions::default())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("large_graph", |b| {
        let traversal = GraphTraversal::new(&large_graph);
        b.iter(|| {
            let result = traversal
                .dfs(black_box("kern_func_0"), kcs_graph::traversal::TraversalOptions::default())
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_path_finding(c: &mut Criterion) {
    let small_graph = create_small_graph();
    let medium_graph = create_medium_graph();
    let large_graph = create_large_graph();

    let mut group = c.benchmark_group("path_finding");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("small_graph_short_path", |b| {
        b.iter(|| {
            let path = small_graph.get_call_path(black_box("func_0"), black_box("func_5"));
            black_box(path);
        });
    });

    group.bench_function("medium_graph_medium_path", |b| {
        b.iter(|| {
            let path = medium_graph.get_call_path(black_box("func_0"), black_box("func_50"));
            black_box(path);
        });
    });

    group.bench_function("large_graph_long_path", |b| {
        b.iter(|| {
            let path =
                large_graph.get_call_path(black_box("kern_func_0"), black_box("kern_func_500"));
            black_box(path);
        });
    });

    group.finish();
}

fn benchmark_cycle_detection(c: &mut Criterion) {
    let small_graph = create_small_graph();
    let medium_graph = create_medium_graph();
    let large_graph = create_large_graph();

    let mut group = c.benchmark_group("cycle_detection");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("small_graph", |b| {
        b.iter(|| {
            let detector = CycleDetector::new(&small_graph);
            let analysis = detector.analyze().unwrap();
            black_box(analysis);
        });
    });

    group.bench_function("medium_graph", |b| {
        b.iter(|| {
            let detector = CycleDetector::new(&medium_graph);
            let analysis = detector.analyze().unwrap();
            black_box(analysis);
        });
    });

    group.bench_function("large_graph", |b| {
        b.iter(|| {
            let detector = CycleDetector::new(&large_graph);
            let analysis = detector.analyze().unwrap();
            black_box(analysis);
        });
    });

    group.finish();
}

fn benchmark_topological_sort(c: &mut Criterion) {
    // Create acyclic graphs for topological sort
    let small_dag = create_small_graph();
    let mut medium_dag = KernelGraph::new();

    // Create a medium DAG (no cycles)
    for i in 0..500 {
        let symbol = Symbol {
            name: format!("dag_func_{}", i),
            file_path: "dag.c".to_string(),
            line_number: (i as u32 + 1) * 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };
        medium_dag.add_symbol(symbol);
    }

    // Create edges that don't form cycles
    for i in 0..499 {
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 1000 + i as u32,
            conditional: false,
            config_guard: None,
        };
        medium_dag
            .add_call(&format!("dag_func_{}", i), &format!("dag_func_{}", i + 1), edge)
            .unwrap();

        // Add forward edges only
        if i % 10 == 0 && i < 490 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 2000 + i as u32,
                conditional: false,
                config_guard: None,
            };
            medium_dag
                .add_call(&format!("dag_func_{}", i), &format!("dag_func_{}", i + 10), edge)
                .unwrap();
        }
    }

    let mut group = c.benchmark_group("topological_sort");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("small_dag", |b| {
        let traversal = GraphTraversal::new(&small_dag);
        b.iter(|| {
            let result = traversal.topological_sort();
            let _ = black_box(result);
        });
    });

    group.bench_function("medium_dag", |b| {
        let traversal = GraphTraversal::new(&medium_dag);
        b.iter(|| {
            let result = traversal.topological_sort();
            let _ = black_box(result);
        });
    });

    group.finish();
}

fn benchmark_filtered_traversal(c: &mut Criterion) {
    let medium_graph = create_medium_graph();

    let mut group = c.benchmark_group("filtered_traversal");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("direct_calls_only", |b| {
        let traversal = GraphTraversal::new(&medium_graph);
        let options = kcs_graph::traversal::TraversalOptions {
            call_type_filter: Some(CallType::Direct),
            ..Default::default()
        };
        b.iter(|| {
            let result = traversal.bfs(black_box("func_0"), options.clone()).unwrap();
            black_box(result);
        });
    });

    group.bench_function("with_depth_limit", |b| {
        let traversal = GraphTraversal::new(&medium_graph);
        let options = kcs_graph::traversal::TraversalOptions {
            max_depth: Some(5),
            ..Default::default()
        };
        b.iter(|| {
            let result = traversal.bfs(black_box("func_0"), options.clone()).unwrap();
            black_box(result);
        });
    });

    group.bench_function("conditional_only", |b| {
        let traversal = GraphTraversal::new(&medium_graph);
        let options = kcs_graph::traversal::TraversalOptions {
            conditional_only: true,
            ..Default::default()
        };
        b.iter(|| {
            let result = traversal.bfs(black_box("func_0"), options.clone()).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_reachability_analysis(c: &mut Criterion) {
    let small_graph = create_small_graph();
    let medium_graph = create_medium_graph();
    let large_graph = create_large_graph();

    let mut group = c.benchmark_group("reachability");
    group.measurement_time(Duration::from_secs(10));

    for depth in [1, 3, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("small_graph", depth), depth, |b, &depth| {
            b.iter(|| {
                let symbols = small_graph.get_reachable_symbols(black_box("func_0"), depth);
                black_box(symbols);
            });
        });

        group.bench_with_input(BenchmarkId::new("medium_graph", depth), depth, |b, &depth| {
            b.iter(|| {
                let symbols = medium_graph.get_reachable_symbols(black_box("func_0"), depth);
                black_box(symbols);
            });
        });

        if *depth <= 5 {
            // Only test smaller depths on large graph
            group.bench_with_input(BenchmarkId::new("large_graph", depth), depth, |b, &depth| {
                b.iter(|| {
                    let symbols =
                        large_graph.get_reachable_symbols(black_box("kern_func_0"), depth);
                    black_box(symbols);
                });
            });
        }
    }

    group.finish();
}

fn benchmark_bidirectional_search(c: &mut Criterion) {
    let small_graph = create_small_graph();
    let medium_graph = create_medium_graph();

    let mut group = c.benchmark_group("bidirectional_search");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("small_graph", |b| {
        let traversal = GraphTraversal::new(&small_graph);
        b.iter(|| {
            let path = traversal.bidirectional_search(black_box("func_0"), black_box("func_15"));
            black_box(path);
        });
    });

    group.bench_function("medium_graph", |b| {
        let traversal = GraphTraversal::new(&medium_graph);
        b.iter(|| {
            let path = traversal.bidirectional_search(black_box("func_0"), black_box("func_100"));
            black_box(path);
        });
    });

    group.finish();
}

// Verify that p95 latency is under 600ms (constitutional requirement)
fn benchmark_p95_compliance(c: &mut Criterion) {
    let large_graph = create_large_graph();

    let mut group = c.benchmark_group("p95_compliance");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(100);

    // Test a realistic query workload
    group.bench_function("who_calls_pattern", |b| {
        b.iter(|| {
            // Simulate finding callers of a function
            let callers = large_graph.find_callers(black_box("kern_func_2500"));
            black_box(callers);
        });
    });

    group.bench_function("list_dependencies_pattern", |b| {
        b.iter(|| {
            // Simulate listing dependencies
            let callees = large_graph.find_callees(black_box("kern_func_100"));
            black_box(callees);
        });
    });

    group.bench_function("trace_execution_pattern", |b| {
        b.iter(|| {
            // Simulate tracing execution path
            let path =
                large_graph.get_call_path(black_box("kern_func_0"), black_box("kern_func_100"));
            black_box(path);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_bfs_traversal,
    benchmark_dfs_traversal,
    benchmark_path_finding,
    benchmark_cycle_detection,
    benchmark_topological_sort,
    benchmark_filtered_traversal,
    benchmark_reachability_analysis,
    benchmark_bidirectional_search,
    benchmark_p95_compliance
);

criterion_main!(benches);
