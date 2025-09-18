# Quick Start: MCP Tools Implementation

This guide demonstrates the actual MCP tool implementations replacing mock data.

## Prerequisites

1. PostgreSQL database running with call graph data
2. KCS MCP server running
3. Valid authentication token

## Setup

```bash
# Start the database (if using Docker)
docker compose up -d postgres

# Run database migrations
make migrate

# Index a kernel repository to populate call graph
tools/index_kernel.sh ~/src/linux

# Start the MCP server
source .venv/bin/activate
kcs-mcp --host 0.0.0.0 --port 8080
```

## Testing the Endpoints

### 1. Who Calls - Find Callers of a Function

Find what calls `vfs_read`:

```bash
curl -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "vfs_read",
    "depth": 2
  }'
```

Expected response:

```json
{
  "callers": [
    {
      "symbol": "ksys_read",
      "span": {
        "path": "fs/read_write.c",
        "sha": "abc123...",
        "start": 621,
        "end": 625
      },
      "call_type": "direct"
    },
    {
      "symbol": "kernel_read",
      "span": {
        "path": "fs/read_write.c",
        "sha": "abc123...",
        "start": 445,
        "end": 450
      },
      "call_type": "direct"
    }
  ]
}
```

### 2. List Dependencies - Find What a Function Calls

Find what `sys_read` calls:

```bash
curl -X POST http://localhost:8080/mcp/tools/list_dependencies \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "sys_read",
    "depth": 1
  }'
```

Expected response:

```json
{
  "callees": [
    {
      "symbol": "ksys_read",
      "span": {
        "path": "fs/read_write.c",
        "sha": "abc123...",
        "start": 634,
        "end": 634
      },
      "call_type": "direct"
    }
  ]
}
```

### 3. Entry Point Flow - Trace System Call Execution

Trace the read system call:

```bash
curl -X POST http://localhost:8080/mcp/tools/entrypoint_flow \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entry": "__NR_read"
  }'
```

Expected response:

```json
{
  "steps": [
    {
      "edge": "syscall",
      "from": "syscall_entry",
      "to": "sys_read",
      "span": {
        "path": "arch/x86/entry/syscalls/syscall_64.tbl",
        "sha": "def456...",
        "start": 10,
        "end": 10
      }
    },
    {
      "edge": "function_call",
      "from": "sys_read",
      "to": "ksys_read",
      "span": {
        "path": "fs/read_write.c",
        "sha": "abc123...",
        "start": 634,
        "end": 634
      }
    },
    {
      "edge": "function_call",
      "from": "ksys_read",
      "to": "vfs_read",
      "span": {
        "path": "fs/read_write.c",
        "sha": "abc123...",
        "start": 621,
        "end": 625
      }
    }
  ]
}
```

### 4. Impact Analysis - Analyze Change Effects

Analyze impact of modifying VFS functions:

```bash
curl -X POST http://localhost:8080/mcp/tools/impact_of \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["vfs_read", "vfs_write"],
    "files": ["fs/read_write.c"]
  }'
```

Expected response:

```json
{
  "configs": ["x86_64:defconfig", "x86_64:allmodconfig"],
  "modules": ["vfs"],
  "tests": ["fs/vfs_test.c"],
  "owners": ["vfs@kernel.org"],
  "risks": ["high_impact_change", "syscall_interface_affected"],
  "cites": [
    {
      "path": "fs/read_write.c",
      "sha": "abc123...",
      "start": 445,
      "end": 625
    }
  ]
}
```

## Validation Tests

### Test 1: Verify Citation Accuracy

Every response must include valid file:line citations:

```bash
# Get a response
response=$(curl -s -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "vfs_read"}')

# Extract and verify a citation
path=$(echo $response | jq -r '.callers[0].span.path')
line=$(echo $response | jq -r '.callers[0].span.start')

# Check the file exists and line is valid
test -f ~/src/linux/$path && echo "✓ File exists"
sed -n "${line}p" ~/src/linux/$path | grep -q "vfs_read" && echo "✓ Citation accurate"
```

### Test 2: Verify Depth Limiting

Deep traversals should be limited:

```bash
time curl -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "memcpy",
    "depth": 5
  }'

# Should complete in < 600ms (constitutional requirement)
```

### Test 3: Verify Empty Results Handling

Non-existent symbols return empty results:

```bash
curl -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "nonexistent_function_xyz"
  }'

# Response: {"callers": []}
```

### Test 4: Verify Configuration Filtering

Results filtered by kernel config:

```bash
curl -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "vfs_read",
    "config": "arm64:defconfig"
  }'

# Returns only callers in ARM64 configuration
```

## Integration Test Suite

Run the full test suite:

```bash
# Run contract tests
pytest tests/contract/test_mcp_tools_contract.py -v

# Run integration tests with real database
pytest tests/integration/test_mcp_tools_integration.py -v

# Run performance benchmarks
pytest tests/performance/test_mcp_tools_performance.py -v
```

## Troubleshooting

### No Results Returned

1. Check database has call graph data:

```sql
SELECT COUNT(*) FROM call_edge;
-- Should be > 0
```

1. Verify symbol exists:

```sql
SELECT * FROM symbol WHERE name = 'vfs_read';
```

1. Check server logs for errors:

```bash
journalctl -u kcs-mcp -f
```

### Slow Queries

1. Check database indexes:

```sql
\d call_edge
-- Should show indexes on caller_id and callee_id
```

1. Monitor query performance:

```sql
EXPLAIN ANALYZE WITH RECURSIVE ...
```

1. Reduce traversal depth if needed

### Authentication Errors

1. Verify token is valid:

```bash
curl http://localhost:8080/health  # Should work without auth
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/mcp/resources
```

1. Check token expiration and refresh if needed

## Performance Expectations

- Simple queries (depth=1): < 100ms
- Deep traversals (depth=5): < 600ms
- Large result sets (100 edges): < 400ms
- Concurrent requests: Support 100 req/s

## Next Steps

1. Index additional kernel configurations
2. Tune database indexes for query patterns
3. Add caching layer if needed
4. Extend entry point mappings beyond syscalls
