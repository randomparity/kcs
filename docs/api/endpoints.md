# API Endpoints

## Overview

The KCS API provides endpoints for kernel analysis, organized into categories:

- **Tools**: MCP tools for kernel analysis (`/mcp/tools/*`)
- **Resources**: MCP resources for data access (`/mcp/resources/*`)  
- **System**: Health and metrics (`/health`, `/metrics`)

## MCP Tools

MCP tools perform kernel analysis operations. **Updated with 5 new infrastructure endpoints for
configuration parsing, specification validation, semantic search, call graph traversal, and
graph export.**

### POST /mcp/tools/diff_spec_vs_code

**Summary**: Check drift between spec and code

**Request Body**:

Content-Type: `application/json`

Properties:

- `feature_id` (string) (required): No description

**Responses**:

- **200**: Drift report
  - Content-Type: `application/json`

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "feature_id": "kcs-sysfs-interface"
}' \
     http://localhost:8080/mcp/tools/diff_spec_vs_code
```text

### POST /mcp/tools/entrypoint_flow

**Summary**: Trace flow from entry point through kernel with support for 40+ syscalls, ioctl, and file_ops

**Request Body**:

Content-Type: `application/json`

Properties:
- `entry` (string) (required): Entry point name (syscall: "__NR_read", ioctl: "IOCTL_*", file_ops: "*_fops")
- `config` (string) (optional): Kernel configuration context

**Responses**:

- **200**: Flow trace with call graph traversal
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "entry": "__NR_read",
  "config": "x86_64:defconfig"
}' \
     http://localhost:8080/mcp/tools/entrypoint_flow
```text

### POST /mcp/tools/generate_reverse_prd

**Summary**: Generate PRD from implementation

**Request Body**:

Content-Type: `application/json`

Properties:
- `entrypoint_id` (string) (optional): No description
- `area` (string) (optional): No description

**Responses**:

- **200**: Generated PRD
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "entrypoint_id": "__NR_read",
  "area": "fs/vfs"
}' \
     http://localhost:8080/mcp/tools/generate_reverse_prd
```text

### POST /mcp/tools/get_symbol

**Summary**: Get symbol information

**Request Body**:

Content-Type: `application/json`

Properties:
- `symbol` (string) (required): No description

**Responses**:

- **200**: Symbol information
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "symbol": "sys_read"
}' \
     http://localhost:8080/mcp/tools/get_symbol
```text

### POST /mcp/tools/impact_of

**Summary**: Analyze impact of changes using bidirectional call graph traversal with subsystem detection

**Request Body**:

Content-Type: `application/json`

Properties:
- `diff` (string) (optional): Git diff to extract symbols from (functions, structs, macros)
- `files` (array) (optional): Files to analyze for impact
- `symbols` (array) (optional): Symbols to analyze for blast radius (callers + callees)
- `config` (string) (optional): Configuration context

**Responses**:

- **200**: Impact analysis with call graph traversal
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "diff": "--- a/fs/read_write.c\n+++ b/fs/read_write.c\n@@ -450,6 +450,7 @@ static ssize_t vfs_read(...)",
  "files": [
    "fs/read_write.c"
  ],
  "symbols": [
    "vfs_read"
  ],
  "config": "x86_64:defconfig"
}' \
     http://localhost:8080/mcp/tools/impact_of
```text

### POST /mcp/tools/list_dependencies

**Summary**: Find functions called by a symbol using recursive call graph traversal with cycle detection

**Request Body**:

Content-Type: `application/json`

Properties:
- `symbol` (string) (required): Function or symbol name to analyze
- `depth` (integer) (optional): Call graph traversal depth (1-5, default: 1)
- `config` (string) (optional): Kernel configuration context

**Responses**:

- **200**: Dependencies with call graph relationships
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "symbol": "sys_read",
  "depth": 2,
  "config": "x86_64:defconfig"
}' \
     http://localhost:8080/mcp/tools/list_dependencies
```text

### POST /mcp/tools/owners_for

**Summary**: Find maintainers

**Request Body**:

Content-Type: `application/json`

Properties:
- `paths` (array) (optional): No description
- `symbols` (array) (optional): No description

**Responses**:

- **200**: Maintainer information
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "paths": [
    "fs/ext4/",
    "fs/btrfs/"
  ],
  "symbols": [
    "ext4_read",
    "btrfs_read"
  ]
}' \
     http://localhost:8080/mcp/tools/owners_for
```text

### POST /mcp/tools/search_code

**Summary**: Search code with semantic or lexical queries

**Request Body**:

Content-Type: `application/json`

Properties:
- `query` (string) (required): Search query
- `topK` (integer) (optional): Maximum results to return

**Responses**:

- **200**: Search results
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "query": "read from file descriptor",
  "topK": 10
}' \
     http://localhost:8080/mcp/tools/search_code
```text

### POST /mcp/tools/search_docs

**Summary**: Search documentation

**Request Body**:

Content-Type: `application/json`

Properties:
- `query` (string) (required): No description
- `corpus` (array) (optional): Document collections to search

**Responses**:

- **200**: Document search results
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "query": "memory barriers",
  "corpus": [
    "Documentation/",
    "Documentation/ABI/"
  ]
}' \
     http://localhost:8080/mcp/tools/search_docs
```text

### POST /mcp/tools/who_calls

**Summary**: Find callers of a symbol using recursive call graph traversal with cycle detection

**Request Body**:

Content-Type: `application/json`

Properties:
- `symbol` (string) (required): Function or symbol name to find callers for
- `depth` (integer) (optional): Call graph traversal depth (1-5, default: 1)
- `config` (string) (optional): Kernel configuration context

**Responses**:

- **200**: Caller information with call graph relationships
  - Content-Type: `application/json`

**Example**:
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "symbol": "vfs_read",
  "depth": 2,
  "config": "x86_64:defconfig"
}' \
     http://localhost:8080/mcp/tools/who_calls
```text

### POST /mcp/tools/parse_kernel_config

**Summary**: Parse kernel configuration file and extract options and dependencies

**Request Body**:

Content-Type: `application/json`

Properties:
- `config_path` (string) (required): Path to kernel .config file
- `arch` (string) (optional): Target architecture (default: x86_64)
- `config_name` (string) (optional): Configuration name identifier (default: custom)
- `resolve_dependencies` (boolean) (optional): Whether to resolve config dependencies
- `incremental` (boolean) (optional): Enable incremental parsing mode
- `base_config_id` (string) (optional): Base config for incremental comparison
- `filters` (object) (optional): Filter options by pattern or subsystem

**Responses**:

- **200**: Parsed configuration data
  - Content-Type: `application/json`

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "config_path": "/usr/src/linux/.config",
  "arch": "x86_64",
  "config_name": "defconfig",
  "resolve_dependencies": true
}' \
     http://localhost:8080/mcp/tools/parse_kernel_config
```text

### POST /mcp/tools/validate_spec

**Summary**: Validate specification against kernel implementation

**Request Body**:

Content-Type: `application/json`

Properties:
- `specification` (object) (required): Specification to validate
  - `name` (string) (required): Specification name
  - `version` (string) (required): Specification version
  - `entrypoint` (string) (required): Main entry point symbol
  - `expected_behavior` (object) (optional): Expected behavior description
  - `parameters` (array) (optional): Expected parameters
- `config` (string) (optional): Kernel configuration context
- `kernel_version` (string) (optional): Target kernel version
- `drift_threshold` (number) (optional): Compliance threshold (0.0-1.0)
- `include_suggestions` (boolean) (optional): Include validation suggestions

**Responses**:

- **200**: Validation results with compliance score
  - Content-Type: `application/json`

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "specification": {
    "name": "sys_read_spec",
    "version": "1.0",
    "entrypoint": "sys_read",
    "expected_behavior": {
      "description": "Read data from file descriptor"
    }
  },
  "drift_threshold": 0.8,
  "include_suggestions": true
}' \
     http://localhost:8080/mcp/tools/validate_spec
```text

### POST /mcp/tools/semantic_search

**Summary**: Perform semantic search on kernel code using embeddings

**Request Body**:

Content-Type: `application/json`

Properties:
- `query` (string) (required): Search query
- `limit` (integer) (optional): Maximum results to return (default: 10)
- `offset` (integer) (optional): Results offset for pagination
- `similarity_threshold` (number) (optional): Minimum similarity score (0.0-1.0)
- `search_mode` (string) (optional): Search mode: semantic, lexical, or hybrid
- `filters` (object) (optional): Search filters by subsystem, file patterns, etc.
- `expand_query` (boolean) (optional): Enable query expansion
- `rerank` (boolean) (optional): Apply result reranking
- `explain` (boolean) (optional): Include search explanations
- `use_cache` (boolean) (optional): Use cached results if available

**Responses**:

- **200**: Semantic search results with similarity scores
  - Content-Type: `application/json`

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "query": "file system read operation",
  "limit": 5,
  "similarity_threshold": 0.7,
  "search_mode": "hybrid",
  "expand_query": true,
  "explain": true
}' \
     http://localhost:8080/mcp/tools/semantic_search
```text

### POST /mcp/tools/traverse_call_graph

**Summary**: Traverse call graph with advanced analysis features including cycle detection and path finding

**Request Body**:

Content-Type: `application/json`

Properties:
- `start_symbol` (string) (required): Starting symbol for traversal
- `direction` (string) (optional): Traversal direction: callers, callees, or both (default: callees)
- `max_depth` (integer) (optional): Maximum traversal depth (1-10, default: 5)
- `detect_cycles` (boolean) (optional): Enable cycle detection
- `find_all_paths` (boolean) (optional): Find all paths to target
- `target_symbol` (string) (optional): Target symbol for path finding
- `filters` (object) (optional): Traversal filters by patterns, subsystems, complexity
- `include_metrics` (boolean) (optional): Include performance metrics
- `include_visualization` (boolean) (optional): Include visualization data
- `layout` (string) (optional): Layout algorithm: hierarchical, force, circular

**Responses**:

- **200**: Call graph with nodes, edges, cycles, and paths
  - Content-Type: `application/json`

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "start_symbol": "sys_read",
  "direction": "callees",
  "max_depth": 3,
  "detect_cycles": true,
  "find_all_paths": true,
  "target_symbol": "vfs_read",
  "include_visualization": true
}' \
     http://localhost:8080/mcp/tools/traverse_call_graph
```text

### POST /mcp/tools/export_graph

**Summary**: Export call graph in various formats with compression and chunking support

**Request Body**:

Content-Type: `application/json`

Properties:
- `root_symbol` (string) (required): Root symbol for graph export
- `format` (string) (required): Export format: json, graphml, dot, or csv
- `depth` (integer) (optional): Graph depth to export (default: 5)
- `filters` (object) (optional): Export filters by patterns, subsystems, edge weights
- `styling` (object) (optional): Visual styling options for DOT/GraphML
- `compress` (boolean) (optional): Enable gzip compression
- `chunk_size` (integer) (optional): Enable chunking with specified size
- `chunk_index` (integer) (optional): Chunk index for paginated export
- `include_metadata` (boolean) (optional): Include node/edge metadata
- `include_annotations` (boolean) (optional): Include annotations
- `include_statistics` (boolean) (optional): Include graph statistics
- `async_export` (boolean) (optional): Process export asynchronously
- `layout` (string) (optional): Layout algorithm for positioning
- `pretty` (boolean) (optional): Pretty-print output

**Responses**:

- **200**: Exported graph data in requested format
  - Content-Type: `application/json`

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
  "root_symbol": "sys_read",
  "format": "graphml",
  "depth": 4,
  "compress": true,
  "include_metadata": true,
  "include_statistics": true,
  "styling": {
    "node_color": "lightblue",
    "edge_color": "gray"
  }
}' \
     http://localhost:8080/mcp/tools/export_graph
```text

## MCP Resources

MCP resources provide access to indexed kernel data.


### GET /mcp/resources

**Summary**: List available MCP resources

**Responses**:

- **200**: Available resources
  - Content-Type: `application/json`

**Example**:
```bash
curl -X GET \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8080/mcp/resources
```text

### GET /mcp/resources/{type}

**Summary**: Access MCP resource

**Parameters**:

- `type` (path, string) (required): No description

**Responses**:

- **200**: Resource content
  - Content-Type: `application/json`

**Example**:
```bash
curl -X GET \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8080/mcp/resources/{type}
```text

## System Endpoints

System endpoints provide health and monitoring information.


### GET /health

**Summary**: Health check

**Responses**:

- **200**: Service healthy
  - Content-Type: `application/json`

**Example**:
```bash
curl -X GET \
     http://localhost:8080/health
```text

### GET /metrics

**Summary**: Service metrics

**Responses**:

- **200**: Prometheus metrics
  - Content-Type: `text/plain`

**Example**:
```bash
curl -X GET \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8080/metrics
```text
