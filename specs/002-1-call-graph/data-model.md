# Data Model: Call Graph Extraction

## Core Entities

### CallEdge

Represents a function call relationship between two functions in kernel source code.

**Fields**:

- `caller`: String - Name of the calling function
- `callee`: String - Name of the called function
- `file_path`: String - Source file path where the call occurs
- `line_number`: u32 - Line number of the call site
- `call_type`: CallType - Type of function call (direct, indirect, macro)

**Relationships**:

- Belongs to a source file (file_path)
- References caller function (by name)
- References callee function (by name)
- Many-to-many: one function can call many others, one function can be called by many

### CallType (Enum)

Categorizes the type of function call for analysis purposes.

**Values**:

- `Direct` - Standard function call: `function_name(args)`
- `Indirect` - Function pointer call: `(*func_ptr)(args)` or `func_ptr(args)`
- `Macro` - Macro invocation that expands to function call

### ParseResult (Extended)

Existing structure enhanced to include call graph data alongside symbol information.

**New Field**:

- `call_edges`: Vec<CallEdge> - Collection of function call relationships found during parsing

**Existing Fields** (unchanged):

- `symbols`: Vec<SymbolInfo> - Function and variable symbols
- `errors`: Vec<ParseError> - Parse errors encountered

## Data Flow

1. **Input**: C source file path
2. **Parser Processing**:
   - Tree-sitter AST generation
   - Symbol extraction (existing)
   - Call edge extraction (new)
3. **Output**: ParseResult with both symbols and call_edges
4. **Storage**: Call edges persisted to PostgreSQL CallEdge table
5. **Query**: MCP endpoints retrieve call relationships for analysis

## Validation Rules

### CallEdge Validation

- `caller` and `callee` must be non-empty strings
- `file_path` must be a valid file path within kernel source
- `line_number` must be > 0
- `call_type` must be a valid enum value

### Data Integrity

- Call edges should reference valid symbol names when possible
- File paths should exist in the indexed kernel repository
- Line numbers should be within file bounds
- No self-referential calls (caller == callee) for direct calls

## State Transitions

Call edges are immutable once extracted - they represent static analysis of source code at a point in time.

**Lifecycle**:

1. **Created**: During source file parsing
2. **Validated**: Structure and content validation
3. **Stored**: Persisted to database with foreign key relationships
4. **Queried**: Retrieved via MCP endpoints for analysis

## Performance Considerations

- Use batch inserts for call edges to optimize database writes
- Index caller and callee columns for fast relationship queries
- Consider partitioning by file_path for large kernel repositories
- Memory usage: ~50 bytes per call edge, expect 100k-1M edges per kernel config
