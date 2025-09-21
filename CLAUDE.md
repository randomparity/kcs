# KCS Call Graph Extraction Capabilities

This document describes the advanced call graph extraction and analysis capabilities available in the Kernel Context Server (KCS).

## Overview

KCS provides comprehensive call graph extraction and analysis for Linux kernel source code through a combination of Tree-sitter AST parsing, Rust-based analysis, and sophisticated graph algorithms. This enables deep understanding of function call relationships, control flow patterns, and architectural dependencies within kernel subsystems.

## Core Capabilities

### 1. Call Graph Extraction (`extract_call_graph`)

Extracts comprehensive call graphs from kernel source files using Tree-sitter parsing and advanced pattern recognition.

**Endpoint**: `POST /extract_call_graph`

**Features**:

- **Direct function calls**: Standard C function invocations
- **Indirect calls**: Function pointer dereferences and callbacks
- **Macro expansions**: Kernel macro call analysis (e.g., `EXPORT_SYMBOL`, `module_init`)
- **Conditional compilation**: Handles `#ifdef` blocks and configuration-dependent code
- **Assembly integration**: Inline assembly call pattern detection
- **Confidence scoring**: Each call edge includes confidence metrics

**Parameters**:

```json
{
  "file_paths": ["path/to/source.c", "path/to/header.h"],
  "include_indirect": true,
  "include_macros": true,
  "max_depth": 10,
  "confidence_threshold": 0.7
}
```

**Response**:

```json
{
  "call_edges": [
    {
      "caller": "sys_read",
      "callee": "vfs_read",
      "file_path": "fs/read_write.c",
      "line_number": 612,
      "call_type": "direct",
      "confidence": 0.95,
      "is_conditional": false,
      "config_guard": null
    }
  ],
  "function_pointers": [...],
  "macro_calls": [...],
  "stats": {
    "total_call_edges": 1247,
    "processing_time_ms": 234
  }
}
```

### 2. Call Relationship Analysis (`get_call_relationships`)

Analyzes the calling relationships for a specific function, finding both callers and callees.

**Endpoint**: `POST /get_call_relationships`

**Features**:

- **Bidirectional analysis**: Find who calls a function and what it calls
- **Depth-limited traversal**: Control analysis scope
- **Configuration context**: Filter by kernel configuration
- **Relationship types**: Direct, indirect, callback, macro-based

**Parameters**:

```json
{
  "function_name": "vfs_read",
  "relationship_type": "both",
  "max_depth": 3,
  "config_context": ["CONFIG_VFS", "CONFIG_BLOCK"]
}
```

**Response**:

```json
{
  "function_name": "vfs_read",
  "callers": [
    {
      "function_name": "sys_read",
      "call_sites": [
        {
          "file_path": "fs/read_write.c",
          "line_number": 612,
          "call_type": "direct",
          "confidence": 0.95
        }
      ]
    }
  ],
  "callees": [
    {
      "function_name": "__vfs_read",
      "call_sites": [...]
    }
  ]
}
```

### 3. Call Path Tracing (`trace_call_path`)

Traces execution paths between two functions through the call graph.

**Endpoint**: `POST /trace_call_path`

**Features**:

- **Multi-path discovery**: Find multiple possible execution paths
- **Path optimization**: Shortest and most likely paths
- **Cycle detection**: Handle recursive calls gracefully
- **Configuration awareness**: Respect kernel build configurations

**Parameters**:

```json
{
  "from_function": "sys_open",
  "to_function": "generic_file_open",
  "max_paths": 5,
  "max_depth": 8,
  "config_context": ["CONFIG_FS"]
}
```

**Response**:

```json
{
  "from_function": "sys_open",
  "to_function": "generic_file_open",
  "paths": [
    {
      "path_id": 1,
      "functions": [
        "sys_open",
        "do_sys_open",
        "do_filp_open",
        "path_openat",
        "generic_file_open"
      ],
      "call_sites": [...],
      "confidence": 0.89,
      "total_depth": 4
    }
  ]
}
```

### 4. Function Pointer Analysis (`analyze_function_pointers`)

Analyzes function pointer usage patterns, callback registrations, and indirect call mechanisms.

**Endpoint**: `POST /analyze_function_pointers`

**Features**:

- **Assignment tracking**: Where function pointers are assigned
- **Callback pattern detection**: Common kernel callback patterns
- **Structure field analysis**: Function pointers in structs
- **Dynamic registration**: Runtime callback registration patterns

**Parameters**:

```json
{
  "file_paths": ["drivers/block/", "fs/"],
  "pointer_patterns": ["file_operations", "block_device_operations"],
  "config_context": ["CONFIG_BLOCK"]
}
```

**Response**:

```json
{
  "function_pointers": [
    {
      "pointer_name": "read",
      "struct_type": "file_operations",
      "assignments": [
        {
          "function_name": "generic_file_read",
          "file_path": "mm/filemap.c",
          "line_number": 2345
        }
      ]
    }
  ],
  "callback_registrations": [...],
  "analysis_stats": {
    "assignments_found": 156,
    "callback_patterns": 23
  }
}
```

## Advanced Features

### Tree-sitter Query Patterns

The system uses sophisticated Tree-sitter queries to detect various call patterns:

- **Direct calls**: `function_name(args)`
- **Function pointers**: `(*func_ptr)(args)`
- **Member calls**: `obj->method(args)`
- **Array calls**: `handlers[i](args)`
- **Macro calls**: `EXPORT_SYMBOL(func)`
- **Kernel-specific patterns**: `module_init()`, `SYSCALL_DEFINE()`

### Configuration-Aware Analysis

Call graph extraction respects kernel configuration:

```c
#ifdef CONFIG_BLOCK
    ret = block_read(buffer);
#else
    ret = simple_read(buffer);
#endif
```

The system tracks conditional compilation blocks and includes configuration guards in the call graph data.

### Performance Optimization

- **Parallel processing**: Multi-threaded analysis for large codebases
- **Incremental updates**: Only re-analyze changed files
- **Caching**: Intelligent caching of AST and call graph data
- **Memory efficiency**: Streaming processing for large kernel trees

## Use Cases

### 1. Vulnerability Analysis

Trace potential attack vectors through the call graph:

```bash
# Find paths from user-controlled syscalls to sensitive kernel functions
curl -X POST /trace_call_path -d '{
  "from_function": "sys_write",
  "to_function": "memcpy",
  "max_paths": 10,
  "max_depth": 15
}'
```

### 2. Impact Analysis

Understand the impact of changing a function:

```bash
# Find all functions affected by modifying vfs_read
curl -X POST /get_call_relationships -d '{
  "function_name": "vfs_read",
  "relationship_type": "callers",
  "max_depth": 5
}'
```

### 3. Architecture Understanding

Map subsystem boundaries and interactions:

```bash
# Extract call graph for the entire VFS subsystem
curl -X POST /extract_call_graph -d '{
  "file_paths": ["fs/"],
  "include_indirect": true,
  "max_depth": 8
}'
```

### 4. Code Quality Analysis

Identify complex call patterns and potential refactoring targets:

```bash
# Analyze function pointer usage in drivers
curl -X POST /analyze_function_pointers -d '{
  "file_paths": ["drivers/"],
  "pointer_patterns": ["*_operations"]
}'
```

## Integration with Analysis Tools

### Graph Visualization

Call graph data can be exported to various formats:

- **DOT format**: For Graphviz visualization
- **JSON**: For custom analysis tools
- **CSV**: For spreadsheet analysis
- **GraphML**: For advanced graph analysis tools

### Static Analysis Integration

The call graph data integrates with:

- **Clang Static Analyzer**: Enhanced path-sensitive analysis
- **Sparse**: Kernel-specific static checking
- **Coccinelle**: Semantic patch development
- **Custom analyzers**: Via JSON API

## Performance Characteristics

### Extraction Speed

- **Small files** (<10KB): ~2-5ms per file
- **Medium files** (10-100KB): ~10-50ms per file
- **Large files** (>100KB): ~50-200ms per file
- **Parallel processing**: 3-5x speedup on multi-core systems

### Memory Usage

- **Baseline overhead**: ~50MB for parser initialization
- **Per-file overhead**: ~1-2MB per analyzed file
- **Graph storage**: ~100 bytes per call edge
- **Peak memory**: Typically <500MB for full kernel analysis

### Scalability

The system can handle:

- **Files**: Up to 10,000 source files per extraction
- **Call edges**: Millions of call relationships
- **Depth**: Analysis depths up to 20 levels
- **Concurrent requests**: Multiple parallel extractions

## Best Practices

### 1. Scoped Analysis

For large codebases, limit scope for better performance:

```json
{
  "file_paths": ["specific/subsystem/"],
  "max_depth": 5,
  "include_indirect": false
}
```

### 2. Incremental Updates

Process only changed files:

```json
{
  "file_paths": ["recently/changed/file.c"],
  "include_indirect": true,
  "max_depth": 3
}
```

### 3. Configuration Context

Always specify relevant configuration context:

```json
{
  "config_context": ["CONFIG_NET", "CONFIG_INET"]
}
```

### 4. Confidence Thresholds

Use appropriate confidence thresholds for your use case:

- **High confidence** (>0.9): Security-critical analysis
- **Medium confidence** (>0.7): General code understanding
- **Low confidence** (>0.5): Exploratory analysis

## Limitations and Considerations

### 1. Dynamic Behavior

Call graph extraction is static analysis and cannot capture:

- Runtime function pointer assignments
- Dynamic module loading effects
- Conditional execution based on runtime data

### 2. Preprocessor Complexity

Complex preprocessor usage may affect accuracy:

- Deeply nested `#ifdef` blocks
- Macro-generated function names
- Token concatenation in macros

### 3. Architecture Dependencies

Some call patterns are architecture-specific:

- Inline assembly calls
- Architecture-specific system calls
- Platform-specific driver interfaces

### 4. Memory and Performance

Large-scale analysis requires consideration of:

- Available system memory
- Processing time for complex codebases
- Network timeouts for large extractions

## Error Handling

The system provides detailed error information:

```json
{
  "error": "extraction_failed",
  "message": "Tree-sitter parsing failed for file.c:123",
  "details": {
    "file_path": "drivers/invalid/file.c",
    "error_type": "syntax_error",
    "line_number": 123,
    "suggestions": ["Check for syntax errors", "Verify file encoding"]
  }
}
```

Common error types:

- **`file_not_found`**: Specified files don't exist
- **`syntax_error`**: Unparseable C code
- **`extraction_timeout`**: Analysis exceeded time limits
- **`memory_limit`**: Insufficient memory for analysis
- **`invalid_config`**: Malformed request parameters

## Future Enhancements

Planned improvements include:

- **Cross-language analysis**: Support for assembly and other languages
- **Dynamic analysis integration**: Runtime call graph data
- **Machine learning**: Pattern recognition for complex call patterns
- **Real-time analysis**: Live analysis during development
- **IDE integration**: Editor plugins for interactive exploration

## Support and Troubleshooting

For issues with call graph extraction:

1. Check file paths and permissions
2. Verify source code syntax
3. Review configuration context
4. Monitor system resources
5. Check KCS logs for detailed error information

The call graph extraction system is designed to provide deep insights into kernel code structure and behavior, enabling advanced analysis, security research, and architectural understanding of complex kernel subsystems.
