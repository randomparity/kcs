# KCS Command Line Interface

This document describes the command-line tools provided by KCS for kernel analysis and indexing.

## Overview

KCS provides several command-line tools:

- **kcs-parser**: Parse C code and extract symbols, call graphs
- **kcs-extractor**: Extract entry points and kernel interfaces
- **kcs-graph**: Analyze call graphs and dependencies
- **kcs-mcp**: Run the MCP protocol server

## kcs-parser

The main parsing tool for extracting symbols and call graphs from C code.

### Usage

```bash
kcs-parser [OPTIONS] <COMMAND> [ARGS...]
```

### Commands

#### `file` - Parse a single file

Parse a single C source file and output analysis results.

```bash
kcs-parser file [OPTIONS] <FILE_PATH>

# Examples
kcs-parser file kernel/sched/core.c
kcs-parser file --include-calls fs/read_write.c
kcs-parser file --format=json --include-calls mm/mmap.c
```

**Options:**

- `--include-calls`: Enable call graph extraction
- `--format=json|ndjson`: Output format (default: json)
- `--output=FILE`: Write output to file instead of stdout
- `--config=CONFIG`: Kernel configuration context

#### `directory` - Parse multiple files

Parse all C files in a directory tree.

```bash
kcs-parser directory [OPTIONS] <DIRECTORY_PATH>

# Examples
kcs-parser directory ~/src/linux/fs/
kcs-parser directory --include-calls --workers=8 ~/src/linux/kernel/
kcs-parser directory --format=ndjson --output=results.ndjson ~/src/linux/mm/
```

**Options:**

- `--include-calls`: Enable call graph extraction for all files
- `--workers=N`: Number of parallel workers (default: 4)
- `--format=json|ndjson`: Output format (default: ndjson for directories)
- `--output=FILE`: Write output to file
- `--filter=GLOB`: Only parse files matching glob pattern

### Global Options

#### `--include-calls`

**Purpose**: Enable call graph extraction and analysis

**Impact**:

- Extracts function call relationships using tree-sitter parsing
- Identifies direct calls, function pointers, and method tables
- Adds call_edges data to output JSON
- Increases parsing time by ~15-20%
- Essential for who_calls, list_dependencies, and impact analysis

**Example Output**:

```json
{
  "file_path": "fs/read_write.c",
  "symbols": [
    {
      "name": "vfs_read",
      "location": {"line": 450, "column": 8},
      "symbol_type": "Function"
    }
  ],
  "call_edges": [
    {
      "caller": "vfs_read",
      "callee": "security_file_permission",
      "call_type": "DirectCall",
      "location": {"line": 453, "column": 8}
    },
    {
      "caller": "vfs_read",
      "callee": "rw_verify_area",
      "call_type": "DirectCall",
      "location": {"line": 456, "column": 6}
    }
  ]
}
```

#### `--format=FORMAT`

**Options**:

- `json`: Pretty-printed JSON (default for single files)
- `ndjson`: Newline-delimited JSON (default for directories)

**When to use**:

- `json`: Human-readable output, small files
- `ndjson`: Streaming processing, large datasets

#### `--config=CONFIG`

**Purpose**: Specify kernel configuration context for conditional compilation

**Format**: `ARCH:CONFIG` (e.g., `x86_64:defconfig`, `arm64:allnoconfig`)

**Example**:

```bash
kcs-parser file --config=x86_64:defconfig --include-calls kernel/sched/core.c
```

### Performance Considerations

#### Memory Usage

Call graph extraction increases memory usage:

| Feature | Memory Impact | Typical Usage |
|---------|---------------|---------------|
| Basic parsing | ~50MB per 1000 files | Baseline |
| + Call graphs | ~75MB per 1000 files | +50% memory |
| + Function pointers | ~100MB per 1000 files | +100% memory |

#### Processing Time

Impact of `--include-calls` on parsing time:

| Code Type | Without Calls | With Calls | Overhead |
|-----------|---------------|------------|----------|
| Simple functions | 0.5ms/file | 0.6ms/file | +20% |
| Complex VFS code | 2.0ms/file | 2.4ms/file | +20% |
| Function pointers | 1.5ms/file | 2.0ms/file | +33% |

#### Scaling

Performance with different directory sizes:

```bash
# Small directories (< 100 files)
kcs-parser directory --include-calls fs/ext4/

# Medium directories (100-1000 files)
kcs-parser directory --include-calls --workers=4 fs/

# Large directories (1000+ files)
kcs-parser directory --include-calls --workers=8 --format=ndjson ~/src/linux/
```

### Integration Examples

#### Database Integration

Parse and populate database:

```bash
# Parse with call graphs and pipe to database loader
kcs-parser directory --include-calls --format=ndjson ~/src/linux/fs/ | \
  python -m kcs_mcp.database_loader --config=x86_64:defconfig

# Direct database integration
export DATABASE_URL="postgresql://user:pass@localhost/kcs"
kcs-parser directory --include-calls --database ~/src/linux/fs/
```

#### CI/CD Integration

Continuous parsing in build pipelines:

```bash
# Parse changed files only
git diff --name-only HEAD~1 | grep '\.c$' | \
  xargs kcs-parser file --include-calls --format=ndjson

# Performance regression testing
kcs-parser directory --include-calls --benchmark ~/src/linux/kernel/ > perf.json
python tools/compare_performance.py baseline.json perf.json
```

#### Incremental Updates

Update only changed files:

```bash
# Find files modified in last day
find ~/src/linux -name "*.c" -mtime -1 | \
  xargs kcs-parser file --include-calls --format=ndjson

# Git-based incremental parsing
git diff --name-only origin/master | grep '\.c$' | \
  xargs kcs-parser file --include-calls
```

## kcs-extractor

Extract kernel entry points and interfaces.

### Basic Usage

```bash
kcs-extractor [OPTIONS] <KERNEL_PATH>

# Examples
kcs-extractor ~/src/linux/
kcs-extractor --syscalls-only ~/src/linux/
kcs-extractor --output=entrypoints.json ~/src/linux/
```

**Options:**

- `--syscalls-only`: Extract only system call entry points
- `--include-ioctls`: Include ioctl interfaces
- `--output=FILE`: Write results to file

## kcs-graph

Analyze call graphs and compute dependencies.

### Graph Usage

```bash
kcs-graph [OPTIONS] <COMMAND>

# Examples
kcs-graph analyze --database-url=$DATABASE_URL
kcs-graph who-calls vfs_read --depth=3
kcs-graph dependencies sys_read --config=x86_64:defconfig
```

**Commands:**

- `analyze`: Compute graph metrics
- `who-calls`: Find callers of a symbol
- `dependencies`: Find callees of a symbol

## kcs-mcp

Run the MCP protocol server.

### Server Usage

```bash
kcs-mcp [OPTIONS]

# Examples
kcs-mcp --host=0.0.0.0 --port=8080
kcs-mcp --workers=8 --log-level=debug
```

**Options:**

- `--host=HOST`: Bind address (default: 127.0.0.1)
- `--port=PORT`: Port number (default: 8080)
- `--workers=N`: Worker processes
- `--log-level=LEVEL`: Logging level

## Environment Variables

All tools respect these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | None |
| `KCS_CONFIG_PATH` | Configuration file path | `~/.kcs/config.toml` |
| `RUST_LOG` | Logging level (trace,debug,info,warn,error) | `info` |
| `KCS_CACHE_DIR` | Cache directory for parsed data | `~/.cache/kcs` |
| `CLANG_PATH` | Path to clang binary | `clang` |

## Common Workflows

### Full Kernel Indexing

Complete kernel analysis with call graphs:

```bash
# 1. Parse all source files with call graphs
kcs-parser directory --include-calls --format=ndjson ~/src/linux/ > parsed.ndjson

# 2. Extract entry points
kcs-extractor ~/src/linux/ > entrypoints.json

# 3. Load into database
export DATABASE_URL="postgresql://kcs:password@localhost/kcs"
python -m kcs_mcp.database_loader parsed.ndjson entrypoints.json

# 4. Start MCP server
kcs-mcp --host=0.0.0.0 --port=8080
```

### Development Workflow

Incremental analysis during development:

```bash
# Parse only modified files
git diff --name-only | grep '\.c$' | \
  xargs kcs-parser file --include-calls --format=json

# Quick symbol lookup
kcs-parser file --include-calls fs/read_write.c | \
  jq '.symbols[] | select(.name=="vfs_read")'

# Test call graph extraction
kcs-parser file --include-calls --format=json kernel/sched/core.c | \
  jq '.call_edges | length'
```

### Performance Analysis

Benchmark parsing performance:

```bash
# Time call graph extraction
time kcs-parser directory --include-calls ~/src/linux/fs/

# Compare with/without call graphs
hyperfine \
  'kcs-parser directory ~/src/linux/kernel/' \
  'kcs-parser directory --include-calls ~/src/linux/kernel/'

# Memory profiling
valgrind --tool=massif kcs-parser directory --include-calls test/
```

## Troubleshooting

### Common Issues

#### 1. Call Graph Extraction Failures

```bash
# Enable verbose logging
RUST_LOG=debug kcs-parser file --include-calls problematic.c

# Check for unsupported syntax
kcs-parser file --include-calls --validate problematic.c

# Skip problematic files
kcs-parser directory --include-calls --skip-errors ~/src/linux/
```

#### 2. Performance Issues

```bash
# Reduce memory usage
kcs-parser directory --include-calls --workers=2 large_directory/

# Process in smaller batches
find large_directory/ -name "*.c" | head -100 | \
  xargs kcs-parser file --include-calls

# Use NDJSON for streaming
kcs-parser directory --include-calls --format=ndjson large_directory/ | \
  head -n 1000 > sample.ndjson
```

#### 3. Database Connection Issues

```bash
# Test database connectivity
psql $DATABASE_URL -c "SELECT 1;"

# Check database schema
kcs-parser file --include-calls --validate-schema test.c

# Reset database
python -m kcs_mcp.database reset_schema
```

### Debug Commands

```bash
# Validate JSON output
kcs-parser file --include-calls test.c | jq empty

# Check call graph extraction
kcs-parser file --include-calls test.c | jq '.call_edges | length'

# Verify symbol extraction
kcs-parser file test.c | jq '.symbols[].name' | sort
```

## API Integration

### Python Integration

```python
import subprocess
import json

def parse_with_calls(file_path):
    result = subprocess.run([
        'kcs-parser', 'file',
        '--include-calls', '--format=json',
        file_path
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

# Usage
data = parse_with_calls('kernel/sched/core.c')
call_edges = data['call_edges']
```

### Shell Integration

```bash
# Function to parse and extract calls
parse_calls() {
    local file="$1"
    kcs-parser file --include-calls --format=json "$file" | \
        jq -r '.call_edges[] | "\(.caller) -> \(.callee)"'
}

# Usage
parse_calls fs/read_write.c | grep vfs_read
```

---

For more information, see:

- [Installation Guide](INSTALLATION.md)
- [Performance Benchmarking](benchmarking.md)
- [API Documentation](api/endpoints.md)
