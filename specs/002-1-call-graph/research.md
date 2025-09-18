# Research: Call Graph Extraction

## Overview

This research phase investigates the best approaches for extracting function call relationships
from C source code using Tree-sitter AST parsing, focusing on the specific needs of Linux kernel
analysis.

## Tree-sitter Call Detection Patterns

### Decision: Use Tree-sitter C Grammar Call Expression Nodes

**Rationale**: Tree-sitter-c provides structured AST nodes specifically for function calls
(`call_expression`), enabling precise call site detection without regex parsing.

**Key Patterns to Match**:

- Direct function calls: `function_name(args)`
- Function pointer calls: `(*func_ptr)(args)` or `func_ptr(args)`
- Macro invocations: Treat as calls when macro expands to function calls
- Member access calls: `obj->method(args)` or `obj.method(args)`

**Alternatives Considered**:

- Regex-based parsing: Rejected due to C preprocessor complexity
- Clang AST integration: Too heavy for initial implementation, kept as future enhancement

## AST Traversal Strategy

### Decision: Recursive Tree-sitter Node Walking

**Rationale**: Tree-sitter provides efficient tree traversal APIs with type-safe node access,
avoiding manual string parsing.

**Implementation Approach**:

1. Walk AST depth-first starting from translation unit
2. For each function_definition node:
   - Extract function name as potential caller
   - Find all call_expression nodes within function body
   - Extract callee names and call site locations
3. Handle edge cases: function pointers, macros, conditional compilation

**Alternatives Considered**:

- Query-based approach using tree-sitter queries: More complex for nested calls
- Breadth-first traversal: Less memory efficient for deep kernel call chains

## Integration with Existing Parser

### Decision: Extend ParseResult with CallEdge Vec

**Rationale**: Minimal disruption to existing symbol extraction while adding call graph data to
the same parsing pass.

**Changes Required**:

- Add `CallEdge` struct with caller, callee, file_path, line_number fields
- Extend `ParseResult` to include `call_edges: Vec<CallEdge>`
- Modify main parsing loop to collect both symbols and call edges
- Update CLI output format to include call graph data

**Alternatives Considered**:

- Separate parsing pass: Inefficient, requires re-parsing same files
- New binary: Adds complexity, breaks unified interface

## Performance Considerations

### Decision: Parallel File Processing with Shared Call Graph

**Rationale**: Kernel parsing is I/O bound, parallel processing of files with thread-safe call
edge collection maximizes throughput.

**Approach**:

- Use existing rayon parallel file processing
- Collect call edges per file, then merge
- No global state during parsing to avoid contention
- Batch database inserts for efficiency

**Performance Targets**:

- Parse call edges at >5k lines/sec
- Memory usage <200MB for typical kernel subsystem
- Incremental parsing when files change

## Database Schema Integration

### Decision: Extend Existing CallEdge Table

**Rationale**: Database schema already exists for call edges, implementation needs to populate
rather than redesign.

**Required Fields** (assuming existing schema):

- caller_symbol_id: Foreign key to symbols table
- callee_symbol_id: Foreign key to symbols table
- file_path: Source file location
- line_number: Call site line number
- call_type: Direct, indirect, macro (optional)

**Database Integration**:

- Use existing PostgreSQL connection patterns
- Batch insert call edges after parsing
- Update foreign key relationships with symbol IDs

## Error Handling Strategy

### Decision: Graceful Degradation with Parse Error Collection

**Rationale**: Kernel code has complex preprocessor usage; parser should capture what it can and
report issues rather than failing completely.

**Error Categories**:

- Unparseable call expressions: Log and skip
- Missing function definitions: Record as external calls
- Macro expansion issues: Mark as macro calls
- Circular dependencies: Detect and break cycles

**Reporting**:

- Include parse errors in ParseResult
- Structured error messages with file:line context
- Distinguish between fatal and recoverable errors

## Testing Strategy

### Decision: Multi-level Test Approach

**Rationale**: Call graph extraction has multiple complexity layers requiring different test strategies.

**Test Levels**:

1. **Unit Tests**: Individual function call patterns
2. **Integration Tests**: Complete C file parsing with known call graphs
3. **Contract Tests**: MCP endpoint responses with call graph data
4. **Performance Tests**: Large kernel subsystem parsing benchmarks

**Test Data**:

- Synthetic C files with known call patterns
- Real kernel subsystem samples (e.g., fs/ext4)
- Edge cases: function pointers, macros, callbacks

## Research Conclusions

All technical unknowns have been resolved. The approach uses:

- Tree-sitter for reliable C AST parsing
- Recursive node traversal for call detection
- Extension of existing ParseResult structure
- Parallel processing for performance
- Graceful error handling for robustness

Implementation can proceed with confidence that this approach will meet the constitutional
requirements for performance, accuracy, and maintainability.
