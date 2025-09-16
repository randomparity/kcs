# Performance Benchmarking

This document describes KCS performance benchmarking capabilities, including call
graph extraction benchmarks and system-wide performance monitoring.

## Call Graph Benchmarks

KCS includes comprehensive benchmarks for call graph extraction performance using
the Criterion benchmarking framework.

### Running Call Graph Benchmarks

```bash
# Run all call graph benchmarks
cargo bench --bench call_graph_bench

# Run with custom configuration
cargo bench --bench call_graph_bench -- --measurement-time 30

# Generate HTML reports (saved to target/criterion/*)
cargo bench --bench call_graph_bench -- --output-format html

# Run specific benchmark groups
cargo bench --bench call_graph_bench -- call_extraction_simple
cargo bench --bench call_graph_bench -- function_pointer_calls
```

### Benchmark Categories

#### 1. Simple Call Extraction

Tests basic function call detection with:

- Direct function calls (`helper1(x)`, `helper2(value)`)
- System calls (`printk()`)
- Simple call chains

**Metrics**:

- Parsing time with/without call extraction
- Memory usage during parsing
- Call detection accuracy

#### 2. Complex Call Patterns

Tests realistic kernel code patterns with:

- VFS file operations (`example_open`, `example_read`)
- System call implementations (`SYSCALL_DEFINE2`)
- Error handling and multiple call paths
- Mutex operations and kernel utilities

**Metrics**:

- Performance with complex call graphs
- Scalability with increasing code complexity
- Memory allocation patterns

#### 3. Function Pointer Detection

Tests advanced call graph features:

- Function pointer arrays (`callbacks[]`)
- Structure-based function pointers (`struct ops`)
- Indirect call detection (`callbacks[index](value)`)
- Method table resolution

**Metrics**:

- Function pointer resolution accuracy
- Performance impact of indirect call analysis
- Memory overhead for pointer tracking

#### 4. Scaling Analysis

Tests performance across different code sizes:

- Generated C code with 10, 50, 100, 200 functions
- Chain call patterns (`func_N` calls `func_N-1`)
- Linear scaling analysis
- Memory usage scaling

**Metrics**:

- Time complexity (O(n) vs O(n²))
- Memory usage per function
- Throughput (functions processed per second)

### Sample Benchmark Code

The benchmarks use realistic kernel code patterns:

```c
// VFS file operations
static int example_open(struct inode *inode, struct file *file) {
    int ret = generic_file_open(inode, file);
    mutex_lock(&example_mutex);
    example_count++;
    mutex_unlock(&example_mutex);
    return ret;
}

// System call implementation
SYSCALL_DEFINE2(example_syscall, int, fd, const char __user *, data) {
    struct file *file = fget(fd);
    int ret = example_validate_data(data);
    ret = example_process_syscall(file, data);
    fput(file);
    return ret;
}
```

### Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Simple calls | < 1ms per file | ✅ Meeting target |
| Complex calls | < 5ms per file | ✅ Meeting target |
| Function pointers | < 10ms per file | ✅ Meeting target |
| Scaling (200 functions) | < 50ms per file | ✅ Meeting target |
| Memory overhead | < 10MB per 1000 functions | ✅ Meeting target |

### Interpreting Results

Benchmark output includes:

```
call_extraction_simple/with_calls
                        time:   [847.23 µs 851.46 µs 856.12 µs]
call_extraction_simple/without_calls
                        time:   [721.34 µs 725.67 µs 730.21 µs]
```

**Key metrics**:

- **Time**: Mean execution time with confidence interval
- **Throughput**: Elements/functions processed per second
- **Overhead**: Performance cost of call graph extraction (typically 15-20%)

### Configuration Options

Benchmark behavior can be customized:

```rust
// Benchmark configuration
let config = ExtendedParserConfig {
    include_call_graphs: true,
    max_call_depth: 10,
    include_function_pointers: true,
    ..Default::default()
};
```

## System Performance Monitoring

### Load Testing with k6

KCS includes k6 load testing scripts for API performance:

```bash
# Run API load tests
make benchmark-k6

# Custom load test configuration
k6 run --vus 10 --duration 30s tests/performance/api_load_test.js

# Stress test with high concurrency
k6 run --vus 100 --duration 2m tests/performance/stress_test.js
```

### Database Performance

Monitor database performance during benchmarks:

```bash
# Check query performance
curl http://localhost:8080/metrics | grep query_duration

# Database statistics
psql $DATABASE_URL -c "
SELECT schemaname,tablename,n_tup_ins,n_tup_upd,n_tup_del
FROM pg_stat_user_tables
WHERE tablename IN ('symbols', 'call_edges', 'files');"

# Index usage analysis
psql $DATABASE_URL -c "
SELECT schemaname,tablename,indexname,idx_scan,idx_tup_read
FROM pg_stat_user_indexes
WHERE tablename = 'call_edges';"
```

### Memory Profiling

Profile memory usage during benchmarks:

```bash
# Run with memory profiling
cargo bench --bench call_graph_bench --features mem-profiling

# Monitor system memory during load tests
htop & make benchmark-k6

# Detailed memory analysis with valgrind (debug builds only)
valgrind --tool=massif target/debug/kcs-parser --include-calls file test.c
```

## Continuous Performance Monitoring

### CI/CD Integration

Benchmarks run automatically in CI:

```yaml
# .github/workflows/benchmarks.yml
- name: Run call graph benchmarks
  run: cargo bench --bench call_graph_bench -- --output-format json

- name: Compare against baseline
  run: python tools/benchmark_comparison.py baseline.json current.json
```

### Performance Regression Detection

Automated detection of performance regressions:

```bash
# Store baseline measurements
cargo bench --bench call_graph_bench -- --save-baseline main

# Compare against baseline
cargo bench --bench call_graph_bench -- --baseline main

# Generate performance report
python tools/generate_performance_report.py
```

### Grafana Dashboards

Monitor performance metrics in Grafana:

- **Call Graph Metrics**: Parse times, memory usage, accuracy
- **API Performance**: Response times, throughput, error rates
- **Database Performance**: Query times, connection pool usage
- **System Resources**: CPU, memory, disk I/O

Access dashboards at: <http://localhost:3000/dashboards>

## Optimization Guidelines

### Code-Level Optimizations

1. **Parser Configuration**
   - Enable call graphs only when needed
   - Set appropriate max call depth
   - Use incremental parsing for updates

2. **Memory Management**
   - Reuse parser instances
   - Clear caches periodically
   - Monitor memory growth

3. **Database Optimization**
   - Use batch inserts for call edges
   - Optimize index usage
   - Regular VACUUM and ANALYZE

### System-Level Optimizations

1. **Parallelization**
   - Use multiple parser workers
   - Parallel database connections
   - Async I/O for file operations

2. **Resource Allocation**
   - Adequate RAM for caching
   - Fast storage for database
   - CPU cores for parsing

3. **Configuration Tuning**
   - Database connection pooling
   - Worker thread counts
   - Cache sizes and TTL

## Troubleshooting Performance Issues

### Common Issues

1. **Slow Call Graph Extraction**

   ```bash
   # Check if complex code patterns
   kcs-parser --include-calls --verbose file slow_file.c

   # Try without function pointers
   kcs-parser --include-calls --no-function-pointers file slow_file.c
   ```

2. **Memory Usage Growth**

   ```bash
   # Monitor memory during parsing
   ./target/debug/kcs-parser --include-calls directory large_kernel/ &
   watch -n 1 'ps aux | grep kcs-parser'

   # Clear caches periodically
   kcs-parser --include-calls --clear-cache directory large_kernel/
   ```

3. **Database Bottlenecks**

   ```bash
   # Check slow queries
   psql $DATABASE_URL -c "
   SELECT query, mean_time, calls
   FROM pg_stat_statements
   WHERE query LIKE '%call_edges%'
   ORDER BY mean_time DESC;"

   # Optimize indexes
   psql $DATABASE_URL -c "REINDEX TABLE call_edges;"
   ```

## Best Practices

1. **Regular Benchmarking**
   - Run benchmarks before releases
   - Track performance over time
   - Set performance budgets

2. **Realistic Testing**
   - Use real kernel code samples
   - Test with various configurations
   - Include edge cases

3. **Comprehensive Metrics**
   - Measure time and memory
   - Track accuracy metrics
   - Monitor system resources

4. **Performance Culture**
   - Profile before optimizing
   - Measure impact of changes
   - Document performance requirements

---

For more detailed performance analysis, see:

- [Database Optimization](sql/optimizations/README.md)
- [Async Performance Guide](docs/async_optimization.md)
- [System Monitoring Setup](docs/monitoring.md)
