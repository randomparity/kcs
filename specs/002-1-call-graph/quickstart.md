# Quickstart: Call Graph Extraction

This guide validates the call graph extraction functionality through practical usage scenarios.

## Prerequisites

- Rust toolchain installed
- KCS development environment setup (`make setup`)
- Sample kernel source files in `tests/fixtures/`

## Test Scenario 1: Basic Call Graph Extraction

**Goal**: Verify that the parser can extract simple function call relationships.

### Setup Test File

Create `tests/fixtures/simple_calls.c`:

```c
int helper_function(int x) {
    return x * 2;
}

int main_function(int a, int b) {
    int result = helper_function(a);
    helper_function(b);
    return result;
}
```

### Run Parser

```bash
# Parse the test file with call graph extraction
kcs-parser parse tests/fixtures/simple_calls.c --format=json --include-calls

# Expected output should include call edges:
# - main_function calls helper_function (line 6)
# - main_function calls helper_function (line 7)
```

### Validation Criteria

- ✅ Parser returns 2 call edges
- ✅ Both calls have caller="main_function", callee="helper_function"
- ✅ Line numbers are correct (6 and 7)
- ✅ Call type is "Direct" for both

## Test Scenario 2: Function Pointer Calls

**Goal**: Verify indirect call detection through function pointers.

### Setup Test File

Create `tests/fixtures/function_pointers.c`:

```c
int operation_a(int x) { return x + 1; }
int operation_b(int x) { return x * 2; }

int execute_operation(int value, int (*op)(int)) {
    return op(value);  // Indirect call
}

int main(void) {
    execute_operation(5, operation_a);
    execute_operation(10, &operation_b);
    return 0;
}
```

### Run Parser

```bash
kcs-parser parse tests/fixtures/function_pointers.c --format=json --include-calls
```

### Validation Criteria

- ✅ Parser detects indirect call in execute_operation (line 5)
- ✅ Call type is "Indirect"
- ✅ Direct calls to execute_operation are also captured

## Test Scenario 3: MCP Endpoint Integration

**Goal**: Verify that MCP endpoints can use call graph data.

### Start MCP Server

```bash
# Ensure database is running
make docker-compose-up-app

# Start MCP server
kcs-mcp --host localhost --port 8080
```

### Index Test Files

```bash
# Index our test files
tools/index_kernel.sh tests/fixtures/
```

### Query Call Relationships

```bash
# Test who_calls endpoint
curl -X POST http://localhost:8080/mcp/who_calls \
  -H "Content-Type: application/json" \
  -d '{"function_name": "helper_function"}'

# Expected: Returns main_function as caller with call sites

# Test list_dependencies endpoint
curl -X POST http://localhost:8080/mcp/list_dependencies \
  -H "Content-Type: application/json" \
  -d '{"function_name": "main_function"}'

# Expected: Returns helper_function as dependency
```

### Validation Criteria

- ✅ who_calls returns correct caller functions
- ✅ list_dependencies returns correct callee functions
- ✅ Responses include file:line citations
- ✅ Call types are correctly identified

## Test Scenario 4: Performance Validation

**Goal**: Verify parser meets performance targets.

### Setup Large Test File

```bash
# Generate test file with many functions and calls
python3 tools/generate_test_kernel.py --functions 1000 --calls-per-function 5 \
  > tests/fixtures/large_kernel.c
```

### Run Performance Test

```bash
# Time the parsing operation
time kcs-parser parse tests/fixtures/large_kernel.c --format=json --include-calls

# Expected: Parse completes in <1 second for 1000 functions
# Expected: Memory usage <50MB during parsing
```

### Validation Criteria

- ✅ Parsing completes in <1 second
- ✅ Memory usage remains under 50MB
- ✅ All call relationships are extracted correctly
- ✅ No parse errors for valid C code

## Test Scenario 5: Error Handling

**Goal**: Verify graceful handling of problematic C code.

### Setup Problematic Test File

Create `tests/fixtures/problematic.c`:

```c
// Incomplete function
int broken_function(

// Macro call
#define CALL_HELPER(x) helper_function(x)

int test_macro(void) {
    CALL_HELPER(42);  // Should be detected as macro call
}

// Function with syntax error
int another_broken() {
    missing_semicolon()
    return 0;
}
```

### Run Parser

```bash
kcs-parser parse tests/fixtures/problematic.c --format=json --include-calls
```

### Validation Criteria

- ✅ Parser doesn't crash on syntax errors
- ✅ Macro call is detected with call_type="Macro"
- ✅ Parse errors are reported in errors array
- ✅ Valid parts of the code are still processed

## Success Criteria

All test scenarios must pass for the feature to be considered complete:

1. **Functional**: Basic call extraction works correctly
2. **Advanced**: Function pointers and indirect calls supported
3. **Integration**: MCP endpoints return call graph data
4. **Performance**: Meets speed and memory targets
5. **Robustness**: Handles errors gracefully

## Next Steps

After completing this quickstart:

1. Run full test suite: `make test`
2. Performance benchmarks: `make test-performance`
3. Integration with real kernel code: `tools/index_kernel.sh ~/linux`
4. Validate all MCP endpoints work with call graph data

This quickstart validates that call graph extraction meets all functional requirements and integrates properly with the existing KCS architecture.
