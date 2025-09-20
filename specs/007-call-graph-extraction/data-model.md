# Data Model: Call Graph Extraction

**Generated**: 2025-01-20
**Status**: Complete

## Overview

Data model design for call graph extraction, defining entities and relationships needed to represent function call patterns in Linux kernel code.

## Core Entities

### CallEdge

Represents a function call relationship between two functions.

**Fields**:
- `caller_id: u64` - Database ID of the calling function
- `callee_id: u64` - Database ID of the called function
- `call_site: CallSite` - Location where the call occurs
- `call_type: CallType` - Classification of the call mechanism
- `confidence: ConfidenceLevel` - Reliability of the call detection
- `conditional: bool` - Whether call is inside conditional compilation
- `config_guard: Option<String>` - Configuration dependency if conditional
- `metadata: CallMetadata` - Additional context information

**Relationships**:
- BelongsTo: Function (caller)
- BelongsTo: Function (callee)
- HasOne: CallSite

**Validation Rules**:
- caller_id != callee_id (no self-loops in direct calls)
- call_site must have valid file path and line number
- config_guard required if conditional = true
- confidence must be High/Medium/Low

### Function

Represents a callable entity in kernel code (extends existing Symbol entity).

**Fields**:
- `id: u64` - Primary key
- `name: String` - Function name
- `signature: Option<String>` - Function signature if available
- `file_path: String` - Source file location
- `line_number: u32` - Line where function is defined
- `symbol_type: SymbolType` - Function/Macro/Variable etc.
- `config_dependencies: Vec<String>` - Required configurations
- `is_callback: bool` - Whether function is used as callback
- `is_entry_point: bool` - Whether function is kernel entry point

**Relationships**:
- HasMany: CallEdge (as caller)
- HasMany: CallEdge (as callee)
- HasMany: FunctionPointer (assignments)

**Validation Rules**:
- name must be valid C identifier
- file_path must exist in indexed files
- line_number > 0
- config_dependencies must be valid kernel configs

### CallSite

Represents the specific location where a function call occurs.

**Fields**:
- `file_path: String` - Source file containing the call
- `line_number: u32` - Line number of the call
- `column_number: Option<u32>` - Column position if available
- `context_before: String` - Code context before the call (3 lines)
- `context_after: String` - Code context after the call (3 lines)
- `function_context: String` - Name of containing function

**Validation Rules**:
- file_path must exist in indexed files
- line_number > 0
- column_number >= 0 if provided
- context strings must not exceed 1000 characters each

### MacroCall

Represents function calls that occur through macro expansion.

**Fields**:
- `macro_name: String` - Name of the macro being expanded
- `macro_definition: String` - Macro definition if available
- `expansion_site: CallSite` - Where macro is used
- `expanded_calls: Vec<CallEdge>` - Function calls in expansion
- `preprocessor_context: String` - Surrounding preprocessor directives

**Relationships**:
- HasMany: CallEdge (expanded calls)
- HasOne: CallSite (expansion site)

**Validation Rules**:
- macro_name must be valid C identifier
- expansion_site must be valid CallSite
- expanded_calls must not be empty

### FunctionPointer

Represents function pointer assignments and their usage patterns.

**Fields**:
- `pointer_name: String` - Name of function pointer variable
- `assignment_site: CallSite` - Where pointer is assigned
- `assigned_function: u64` - Function ID being assigned
- `usage_sites: Vec<CallSite>` - Where pointer is called
- `struct_context: Option<String>` - Struct name if pointer is member

**Relationships**:
- BelongsTo: Function (assigned function)
- HasMany: CallSite (usage sites)

**Validation Rules**:
- pointer_name must be valid C identifier
- assigned_function must exist in Function table
- usage_sites must be valid CallSite objects

### CallPath

Represents a sequence of function calls from entry point to target.

**Fields**:
- `entry_point: u64` - Starting function ID
- `target_function: u64` - Ending function ID
- `path_edges: Vec<u64>` - Sequence of CallEdge IDs forming path
- `path_length: u32` - Number of hops in path
- `total_confidence: f32` - Combined confidence of all edges
- `config_context: String` - Required configuration for path

**Relationships**:
- BelongsTo: Function (entry point)
- BelongsTo: Function (target)
- ReferencesMany: CallEdge (path edges)

**Validation Rules**:
- entry_point != target_function
- path_edges must form continuous path
- path_length must equal path_edges.len()
- total_confidence between 0.0 and 1.0

## Enumerations

### CallType

Classification of function call mechanisms:

```rust
pub enum CallType {
    Direct,        // Standard function call: foo()
    Indirect,      // Function pointer call: (*ptr)()
    Macro,         // Macro expansion call: MACRO()
    Callback,      // Registered callback: ops->handler()
    Conditional,   // Conditional compilation call: #ifdef foo()
    Assembly,      // Inline assembly call
    Syscall,       // System call entry point
}
```

### ConfidenceLevel

Reliability of call detection:

```rust
pub enum ConfidenceLevel {
    High,      // Direct call, clear target
    Medium,    // Function pointer with known assignment
    Low,       // Indirect call, multiple possible targets
}
```

### CallMetadata

Additional context for call relationships:

```rust
pub struct CallMetadata {
    pub call_arguments: Option<String>,
    pub return_value_used: bool,
    pub error_handling_present: bool,
    pub recursion_depth: Option<u32>,
    pub analysis_timestamp: DateTime<Utc>,
}
```

## Database Schema Extensions

### New Tables

**call_edges**:
```sql
CREATE TABLE call_edges (
    id BIGSERIAL PRIMARY KEY,
    caller_id BIGINT REFERENCES symbols(id),
    callee_id BIGINT REFERENCES symbols(id),
    file_path VARCHAR(1000) NOT NULL,
    line_number INTEGER NOT NULL,
    column_number INTEGER,
    call_type VARCHAR(20) NOT NULL,
    confidence VARCHAR(10) NOT NULL,
    conditional BOOLEAN DEFAULT FALSE,
    config_guard VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**function_pointers**:
```sql
CREATE TABLE function_pointers (
    id BIGSERIAL PRIMARY KEY,
    pointer_name VARCHAR(255) NOT NULL,
    assigned_function_id BIGINT REFERENCES symbols(id),
    assignment_file VARCHAR(1000) NOT NULL,
    assignment_line INTEGER NOT NULL,
    struct_context VARCHAR(255),
    usage_sites JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**macro_calls**:
```sql
CREATE TABLE macro_calls (
    id BIGSERIAL PRIMARY KEY,
    macro_name VARCHAR(255) NOT NULL,
    macro_definition TEXT,
    expansion_file VARCHAR(1000) NOT NULL,
    expansion_line INTEGER NOT NULL,
    expanded_call_ids BIGINT[],
    preprocessor_context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes for Performance

```sql
-- Call graph traversal
CREATE INDEX idx_call_edges_caller ON call_edges(caller_id);
CREATE INDEX idx_call_edges_callee ON call_edges(callee_id);
CREATE INDEX idx_call_edges_type ON call_edges(call_type);
CREATE INDEX idx_call_edges_config ON call_edges(config_guard);

-- Location-based queries
CREATE INDEX idx_call_edges_file ON call_edges(file_path);
CREATE INDEX idx_call_edges_location ON call_edges(file_path, line_number);

-- Function analysis
CREATE INDEX idx_function_pointers_name ON function_pointers(pointer_name);
CREATE INDEX idx_function_pointers_assigned ON function_pointers(assigned_function_id);

-- Macro analysis
CREATE INDEX idx_macro_calls_name ON macro_calls(macro_name);
CREATE INDEX idx_macro_calls_file ON macro_calls(expansion_file);
```

## State Transitions

### Call Edge Lifecycle

1. **Discovered**: Call relationship detected in AST
2. **Classified**: Call type determined (direct/indirect/macro/callback)
3. **Validated**: Target function confirmed to exist
4. **Stored**: Persisted to database with metadata
5. **Indexed**: Available for query through MCP endpoints

### Function Pointer Lifecycle

1. **Assignment Detected**: Function assigned to pointer variable
2. **Usage Tracked**: Call sites through pointer identified
3. **Linked**: Assignment connected to usage sites
4. **Confidence Scored**: Reliability assessment completed

## Integration Points

### Existing Symbol Table

Call graph extraction extends the existing symbol table without breaking changes:

- Reuses existing `symbols` table for Function entities
- Links call_edges to existing symbol IDs
- Maintains existing file_path and line_number conventions
- Preserves configuration dependency tracking

### MCP Protocol Integration

Data model maps to MCP endpoints:

- `who_calls` → Query call_edges where callee_id = target
- `list_dependencies` → Query call_edges where caller_id = source
- `entrypoint_flow` → Traverse call_edges from entry points
- `impact_of` → Find all paths through call_edges graph

### Performance Considerations

- Batch insertion for large call graphs
- Prepared statements for relationship queries
- Connection pooling for concurrent access
- Index optimization for graph traversal patterns

**Data model complete and ready for contract generation**