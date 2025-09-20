# Quickstart: Call Graph Extraction

**Feature**: Call Graph Extraction Specification
**Generated**: 2025-01-20

## Prerequisites

- KCS development environment set up
- PostgreSQL with pgvector extension running
- Linux kernel source code available
- Rust 1.75+ and Python 3.11+ installed

## Quick Test Scenarios

### Scenario 1: Extract Direct Function Calls

**Objective**: Verify basic call graph extraction from kernel source files

**Steps**:
1. Start KCS MCP server:
   ```bash
   source .venv/bin/activate
   kcs-mcp --host 0.0.0.0 --port 8080
   ```

2. Extract call graph from test file:
   ```bash
   curl -X POST http://localhost:8080/mcp/tools/extract_call_graph \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
     -d '{
       "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
       "include_indirect": true,
       "include_macros": true,
       "config_context": "x86_64:defconfig"
     }'
   ```

**Expected Result**:
```json
{
  "call_edges": [
    {
      "caller": {
        "name": "test_open",
        "file_path": "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
        "line_number": 15
      },
      "callee": {
        "name": "generic_file_open",
        "file_path": "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
        "line_number": 8
      },
      "call_site": {
        "file_path": "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
        "line_number": 17,
        "function_context": "test_open"
      },
      "call_type": "direct",
      "confidence": "high",
      "conditional": false
    }
  ],
  "extraction_stats": {
    "files_processed": 1,
    "functions_analyzed": 2,
    "call_edges_found": 1,
    "processing_time_ms": 150
  }
}
```

### Scenario 2: Query Function Relationships

**Objective**: Test call relationship queries for existing functions

**Steps**:
1. Query who calls a specific function:
   ```bash
   curl -X POST http://localhost:8080/mcp/tools/get_call_relationships \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
     -d '{
       "function_name": "generic_file_open",
       "relationship_type": "callers",
       "config_context": "x86_64:defconfig",
       "max_depth": 2
     }'
   ```

**Expected Result**:
```json
{
  "function_name": "generic_file_open",
  "relationships": {
    "callers": [
      {
        "function": {
          "name": "test_open",
          "file_path": "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
          "line_number": 15
        },
        "call_edge": {
          "call_type": "direct",
          "confidence": "high",
          "call_site": {
            "line_number": 17
          }
        },
        "depth": 1
      }
    ]
  }
}
```

### Scenario 3: Trace Call Path

**Objective**: Verify call path tracing between functions

**Steps**:
1. Trace path between entry point and target function:
   ```bash
   curl -X POST http://localhost:8080/mcp/tools/trace_call_path \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
     -d '{
       "from_function": "sys_open",
       "to_function": "generic_file_open",
       "config_context": "x86_64:defconfig",
       "max_paths": 3,
       "max_depth": 5
     }'
   ```

**Expected Result**:
```json
{
  "from_function": "sys_open",
  "to_function": "generic_file_open",
  "paths": [
    {
      "path_edges": [
        {
          "caller": {"name": "sys_open"},
          "callee": {"name": "do_sys_open"},
          "call_type": "direct",
          "confidence": "high"
        },
        {
          "caller": {"name": "do_sys_open"},
          "callee": {"name": "generic_file_open"},
          "call_type": "direct",
          "confidence": "high"
        }
      ],
      "path_length": 2,
      "total_confidence": 1.0,
      "config_context": "x86_64:defconfig"
    }
  ]
}
```

### Scenario 4: Analyze Function Pointers

**Objective**: Test function pointer assignment and usage detection

**Steps**:
1. Analyze function pointer patterns:
   ```bash
   curl -X POST http://localhost:8080/mcp/tools/analyze_function_pointers \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
     -d '{
       "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
       "pointer_patterns": ["file_operations"],
       "config_context": "x86_64:defconfig"
     }'
   ```

**Expected Result**:
```json
{
  "function_pointers": [
    {
      "pointer_name": "test_fops",
      "assignment_site": {
        "file_path": "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
        "line_number": 25
      },
      "assigned_function": {
        "name": "test_open",
        "file_path": "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
        "line_number": 15
      },
      "struct_context": "file_operations"
    }
  ],
  "analysis_stats": {
    "pointers_analyzed": 1,
    "assignments_found": 1,
    "usage_sites_found": 0,
    "callback_patterns_matched": 1
  }
}
```

## Integration Testing

### Test File Setup

Create test kernel source file at `tests/fixtures/mini-kernel-v6.1/fs/test_file.c`:

```c
#include <linux/fs.h>
#include <linux/kernel.h>

// Simple function for testing direct calls
static int generic_file_open(struct inode *inode, struct file *file)
{
    return 0;
}

// Function that makes direct calls
static int test_open(struct inode *inode, struct file *file)
{
    return generic_file_open(inode, file);
}

// Function pointer assignment for callback testing
static const struct file_operations test_fops = {
    .open = test_open,
    .read = generic_file_read,
};

// Macro call example
#define CALL_HELPER(func, arg) func(arg)

static int test_macro_call(int value)
{
    return CALL_HELPER(some_helper, value);
}
```

### Performance Verification

Run performance test on larger kernel subsystem:

```bash
time curl -X POST http://localhost:8080/mcp/tools/extract_call_graph \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
  -d '{
    "file_paths": ["fs/ext4/*.c"],
    "include_indirect": true,
    "include_macros": true,
    "config_context": "x86_64:defconfig"
  }'
```

**Performance Targets**:
- Single file extraction: < 200ms
- Small subsystem (10 files): < 2 seconds
- Large subsystem (100 files): < 20 seconds
- Response time p95: < 600ms for queries

## Validation Checklist

### Functional Requirements Validation

- [ ] **FR-001**: Direct function calls extracted with citations ✓
- [ ] **FR-002**: Function pointer assignments tracked ✓
- [ ] **FR-003**: Macro calls expanded and analyzed ✓
- [ ] **FR-004**: Callback patterns recognized ✓
- [ ] **FR-005**: Call sites include precise location data ✓
- [ ] **FR-006**: Conditional compilation context captured ✓
- [ ] **FR-007**: Performance targets met (20min for 50k files) ✓
- [ ] **FR-008**: Integration with Tree-sitter preserved ✓
- [ ] **FR-009**: Call types classified correctly ✓
- [ ] **FR-010**: Cross-file calls handled ✓
- [ ] **FR-011**: Call depth analysis provided ✓
- [ ] **FR-012**: MCP endpoints respond sub-second ✓

### Edge Case Testing

Test edge cases from specification:

1. **Deeply nested function pointers**:
   ```c
   struct ops {
       int (*callback)(void);
   };
   struct container {
       struct ops *ops_ptr;
   };
   // Call: container->ops_ptr->callback()
   ```

2. **Multi-expansion macros**:
   ```c
   #define MULTI_CALL(a, b) do { func1(a); func2(b); } while(0)
   ```

3. **Assembly inline blocks**:
   ```c
   static inline void cpu_relax(void)
   {
       asm volatile("rep; nop" ::: "memory");
   }
   ```

4. **Recursive calls**:
   ```c
   int recursive_func(int n) {
       if (n <= 1) return 1;
       return n * recursive_func(n - 1);
   }
   ```

## Troubleshooting

### Common Issues

1. **No call edges found**:
   - Verify Tree-sitter C grammar is loaded
   - Check file paths are accessible
   - Confirm functions exist in symbol table

2. **Low confidence scores**:
   - Enable clang integration for better analysis
   - Verify compile_commands.json exists
   - Check configuration context matches build

3. **Missing function pointers**:
   - Verify pointer pattern matching rules
   - Check struct definitions are parsed
   - Confirm assignment sites are detected

4. **Performance issues**:
   - Enable parallel processing
   - Use incremental analysis mode
   - Check database index performance

### Debug Commands

Enable verbose logging:
```bash
export RUST_LOG=debug
export LOG_LEVEL=DEBUG
kcs-mcp --host 0.0.0.0 --port 8080
```

Test individual components:
```bash
# Test AST parsing
kcs-parser --include-calls file tests/fixtures/mini-kernel-v6.1/fs/test_file.c

# Test call extraction
kcs-extractor --call-edges tests/fixtures/mini-kernel-v6.1/fs/test_file.c

# Test graph storage
kcs-graph --load-calls extracted_calls.json
```

**Quickstart validation complete** - System ready for full implementation