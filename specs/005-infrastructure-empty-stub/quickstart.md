# Infrastructure Components Quickstart

**Feature**: Infrastructure Core Components Implementation
**Purpose**: Validate all infrastructure components work together

## Prerequisites

1. KCS server running locally:

   ```bash
   source .venv/bin/activate
   kcs-mcp --host 0.0.0.0 --port 8080
   ```

2. PostgreSQL with pgvector extension:

   ```bash
   docker compose up postgres
   ```

3. Sample kernel repository:

   ```bash
   git clone https://github.com/torvalds/linux ~/test-kernel
   cd ~/test-kernel
   git checkout v6.11
   ```

## Test Scenarios

### 1. Multi-Architecture Config Parsing

Parse configurations for different architectures:

```bash
# Parse x86_64 default config
curl -X POST http://localhost:8080/mcp/tools/parse_kernel_config \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "arch/x86/configs/x86_64_defconfig",
    "kernel_root": "~/test-kernel",
    "architecture": "x86_64",
    "config_type": "defconfig"
  }'

# Expected: Config parsed with 1000+ features enabled

# Parse ARM64 config
curl -X POST http://localhost:8080/mcp/tools/parse_kernel_config \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "arch/arm64/configs/defconfig",
    "kernel_root": "~/test-kernel",
    "architecture": "arm64",
    "config_type": "defconfig"
  }'

# Expected: Different feature set than x86_64
```

### 2. Spec Validation

Create a simple spec and validate implementation:

```bash
# Create spec file
cat > /tmp/vfs_spec.md << 'EOF'
# VFS Specification

## Requirements

### FR-001: File Operations
System MUST provide sys_open syscall for file opening.

### FR-002: Permission Checking
System MUST check file permissions before allowing access.

### FR-003: Error Handling
System MUST return -ENOENT when file does not exist.
EOF

# Validate against kernel
curl -X POST http://localhost:8080/mcp/tools/validate_spec \
  -H "Content-Type: application/json" \
  -d '{
    "spec_content": "'$(cat /tmp/vfs_spec.md | jq -Rs .)'",
    "spec_format": "markdown",
    "kernel_commit": "HEAD",
    "config_name": "x86_64:defconfig",
    "focus_areas": ["fs"]
  }'

# Expected output:
# {
#   "report_id": "uuid-here",
#   "total_requirements": 3,
#   "passed": 3,
#   "failed": 0,
#   "unknown": 0,
#   "severity": "info"
# }
```

### 3. Semantic Search

Test natural language search capabilities:

```bash
# Search for memory allocation concepts
curl -X POST http://localhost:8080/mcp/tools/semantic_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does kernel allocate memory for processes",
    "query_type": "concept",
    "context": "mm",
    "limit": 5
  }'

# Expected: Results include kmalloc, __get_free_pages, etc.

# Find similar functions
curl -X POST http://localhost:8080/mcp/tools/semantic_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "functions similar to mutex_lock",
    "query_type": "similar_to",
    "limit": 10
  }'

# Expected: Results include spin_lock, rw_lock, semaphore functions
```

### 4. Call Graph Traversal

Trace call paths with cycle detection:

```bash
# Find path from syscall to implementation
curl -X POST http://localhost:8080/mcp/tools/traverse_call_graph \
  -H "Content-Type: application/json" \
  -d '{
    "source": "sys_read",
    "target": "generic_file_read",
    "direction": "forward",
    "max_depth": 5,
    "detect_cycles": true
  }'

# Expected: Multiple paths showing call chain

# Detect recursive patterns
curl -X POST http://localhost:8080/mcp/tools/traverse_call_graph \
  -H "Content-Type: application/json" \
  -d '{
    "source": "shrink_dcache_parent",
    "direction": "forward",
    "max_depth": 10,
    "detect_cycles": true
  }'

# Expected: has_cycles=true if recursion detected
```

### 5. Graph Export

Export subgraph for visualization:

```bash
# Export VFS subsystem graph
curl -X POST http://localhost:8080/mcp/tools/export_graph \
  -H "Content-Type: application/json" \
  -d '{
    "format": "json_graph",
    "scope": "subgraph",
    "root_function": "vfs_open",
    "max_depth": 3,
    "chunk_size_mb": 50
  }'

# Expected output:
# {
#   "export_id": "uuid-here",
#   "format": "json_graph",
#   "node_count": 150,
#   "edge_count": 280,
#   "file_size_bytes": 2500000,
#   "chunk_count": 1,
#   "chunks": [...]
# }

# Download and verify with jq
curl -o graph.json "chunk_url_from_response"
jq '.nodes | length' graph.json  # Should match node_count
```

### 6. Integration Test

Full workflow combining all components:

```bash
# 1. Parse ARM64 config
CONFIG_RESPONSE=$(curl -s -X POST http://localhost:8080/mcp/tools/parse_kernel_config \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "arch/arm64/configs/defconfig",
    "kernel_root": "~/test-kernel",
    "architecture": "arm64"
  }')

CONFIG_NAME=$(echo $CONFIG_RESPONSE | jq -r '.config_name')
echo "Parsed config: $CONFIG_NAME"

# 2. Search for ARM64-specific code
SEARCH_RESPONSE=$(curl -s -X POST http://localhost:8080/mcp/tools/semantic_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ARM64 exception handling",
    "config_filter": "'$CONFIG_NAME'",
    "limit": 5
  }')

FIRST_FUNCTION=$(echo $SEARCH_RESPONSE | jq -r '.results[0].symbol')
echo "Found ARM64 function: $FIRST_FUNCTION"

# 3. Trace its call graph
GRAPH_RESPONSE=$(curl -s -X POST http://localhost:8080/mcp/tools/traverse_call_graph \
  -H "Content-Type: application/json" \
  -d '{
    "source": "'$FIRST_FUNCTION'",
    "direction": "both",
    "max_depth": 3,
    "config_name": "'$CONFIG_NAME'"
  }')

echo "Found $(echo $GRAPH_RESPONSE | jq '.paths_found') call paths"

# 4. Export for analysis
EXPORT_RESPONSE=$(curl -s -X POST http://localhost:8080/mcp/tools/export_graph \
  -H "Content-Type: application/json" \
  -d '{
    "format": "graphml",
    "scope": "subgraph",
    "root_function": "'$FIRST_FUNCTION'",
    "config_name": "'$CONFIG_NAME'"
  }')

echo "Exported $(echo $EXPORT_RESPONSE | jq '.node_count') nodes"
```

## Validation Checklist

- [ ] All architectures parse successfully (x86, ARM64, RISC-V)
- [ ] Spec validation identifies violations correctly
- [ ] Semantic search returns relevant results
- [ ] Call graph traversal finds valid paths
- [ ] Cycle detection works for recursive functions
- [ ] Graph export produces valid, parseable files
- [ ] Files under 1GB are jq-compatible
- [ ] Performance meets targets (queries <600ms)
- [ ] Config-aware analysis shows different results per architecture
- [ ] Integration test completes successfully

## Troubleshooting

### Config parsing fails

- Check kernel source path is correct
- Verify Kconfig files exist for architecture
- Check config_type matches actual file

### Semantic search returns no results

- Verify pgvector extension is enabled
- Check embeddings are generated (may take time on first run)
- Try broader search terms

### Graph traversal times out

- Reduce max_depth parameter
- Use more specific source/target functions
- Enable config filtering to reduce search space

### Export files too large

- Reduce chunk_size_mb parameter
- Use subgraph scope instead of full
- Increase max_depth gradually

## Performance Benchmarks

Run these after implementation to verify constitutional requirements:

```bash
# Query performance (must be <600ms p95)
for i in {1..100}; do
  time curl -s -X POST http://localhost:8080/mcp/tools/semantic_search \
    -d '{"query": "test query '$i'"}' > /dev/null
done | grep real | sort -n | tail -5

# Memory usage during large export
/usr/bin/time -v curl -X POST http://localhost:8080/mcp/tools/export_graph \
  -d '{"format": "json_graph", "scope": "full"}' > /dev/null 2>&1 | grep "Maximum resident"

# Should be under 16GB for 50k+ files
```

## Next Steps

After validating all components:

1. Run integration tests: `pytest tests/integration/test_infrastructure.py`
2. Run performance tests: `pytest tests/performance/test_infra_perf.py`
3. Update MCP documentation with new endpoints
4. Create visualization tools for exported graphs
5. Set up monitoring for embedding generation
