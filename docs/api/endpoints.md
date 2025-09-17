# API Endpoints

## Overview

The KCS API provides endpoints for kernel analysis, organized into categories:

- **Tools**: MCP tools for kernel analysis (`/mcp/tools/*`)
- **Resources**: MCP resources for data access (`/mcp/resources/*`)  
- **System**: Health and metrics (`/health`, `/metrics`)

## MCP Tools

MCP tools perform kernel analysis operations.

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
