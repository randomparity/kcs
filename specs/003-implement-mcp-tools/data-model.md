# Data Model: MCP Tools Implementation

**Feature**: Implement MCP Tools
**Date**: 2025-09-16

## Entities

### CallEdge (Existing)

Represents a function call relationship in the kernel.

**Fields**:
- `caller_id`: Foreign key to Symbol table
- `callee_id`: Foreign key to Symbol table
- `call_type`: Enum (direct, indirect, macro)
- `line_number`: Line where call occurs
- `config_bitmap`: Configuration where edge exists

**Relationships**:
- Many-to-one with Symbol (caller)
- Many-to-one with Symbol (callee)

**Validation**:
- caller_id and callee_id must be different (no self-calls recorded)
- line_number must be positive
- call_type must be valid enum value

### Symbol (Existing)

Represents a kernel function, variable, or type.

**Fields**:
- `id`: Primary key
- `name`: Symbol name (e.g., "vfs_read")
- `kind`: Type of symbol (function, variable, etc.)
- `file_id`: Foreign key to File table
- `start_line`: Beginning line number
- `end_line`: Ending line number
- `signature`: Function signature if applicable
- `config_bitmap`: Configurations where symbol exists

**Relationships**:
- Many-to-one with File
- One-to-many with CallEdge (as caller)
- One-to-many with CallEdge (as callee)

**Validation**:
- name must be non-empty
- start_line <= end_line
- Both line numbers positive

### File (Existing)

Represents a source file in the kernel.

**Fields**:
- `id`: Primary key
- `path`: Relative path from kernel root
- `sha`: Content hash for version tracking
- `last_parsed`: Timestamp of last analysis

**Relationships**:
- One-to-many with Symbol

**Validation**:
- path must be unique per configuration
- sha must be valid hex string

### Span (Response Model)

Citation information for source location.

**Fields**:
- `path`: File path relative to kernel root
- `sha`: File content hash
- `start`: Starting line number
- `end`: Ending line number

**Validation**:
- All fields required
- start <= end
- Line numbers positive

### CallerInfo (Response Model)

Information about a calling or called function.

**Fields**:
- `symbol`: Function name
- `span`: Source location
- `call_type`: Type of call (direct, indirect, macro)

**Validation**:
- All fields required
- call_type must be valid enum

### FlowStep (Response Model)

Single step in execution flow trace.

**Fields**:
- `edge`: Edge type (syscall, function_call, etc.)
- `from`: Source symbol
- `to`: Target symbol
- `span`: Source location of call

**Validation**:
- All fields required
- from and to must be different

### ImpactResult (Response Model)

Impact analysis results.

**Fields**:
- `configs`: Affected kernel configurations
- `modules`: Affected kernel modules
- `tests`: Relevant test files
- `owners`: Maintainer contacts
- `risks`: Risk assessment flags
- `cites`: Source citations for evidence

**Validation**:
- All arrays can be empty but must be present
- cites must contain valid Span objects

## State Transitions

### Call Graph Traversal States

```
INIT -> TRAVERSING -> COMPLETE
        ↓
     CYCLE_DETECTED -> COMPLETE
```

- **INIT**: Starting traversal from symbol
- **TRAVERSING**: Following edges, tracking visited
- **CYCLE_DETECTED**: Found circular dependency
- **COMPLETE**: Reached depth limit or no more edges

### Database Query States

```
PENDING -> EXECUTING -> SUCCESS
           ↓
        ERROR -> EMPTY_RESULT
```

- **PENDING**: Query prepared
- **EXECUTING**: Running against database
- **SUCCESS**: Results returned
- **ERROR**: Query failed (timeout, connection issue)
- **EMPTY_RESULT**: Graceful degradation

## Business Rules

### Traversal Rules

1. **Depth Limiting**: Maximum traversal depth of 5 to prevent performance issues
2. **Cycle Prevention**: Track visited symbols, stop on revisit
3. **Result Limiting**: Maximum 100 results per query for performance
4. **Config Filtering**: Optional configuration filter on all queries

### Citation Rules

1. **Required Citations**: Every result must include file:line span
2. **Valid References**: Citations must point to actual file locations
3. **SHA Tracking**: Include file SHA for version verification

### Impact Analysis Rules

1. **Bidirectional Search**: Check both callers and callees
2. **Subsystem Detection**: Identify subsystem from symbol prefix patterns
3. **Risk Assessment**: Flag high risk if >10 affected symbols
4. **Syscall Detection**: Flag syscall interface changes specially

## Query Patterns

### Find Callers Pattern
```sql
WITH RECURSIVE caller_tree AS (
    -- Base case: direct callers
    SELECT caller_id, callee_id, call_type, 1 as depth
    FROM call_edge
    WHERE callee_id = (SELECT id FROM symbol WHERE name = $1)

    UNION ALL

    -- Recursive case: callers of callers
    SELECT e.caller_id, e.callee_id, e.call_type, ct.depth + 1
    FROM call_edge e
    JOIN caller_tree ct ON e.callee_id = ct.caller_id
    WHERE ct.depth < $2  -- depth limit
)
SELECT DISTINCT ... FROM caller_tree
```

### Find Callees Pattern
```sql
WITH RECURSIVE callee_tree AS (
    -- Similar structure but following edges forward
    ...
)
SELECT DISTINCT ... FROM callee_tree
```

### Entry Point Flow Pattern
```sql
-- Map entry point to initial function
-- Then follow callee chain with depth limit
-- Track visited to prevent cycles
```

## Performance Considerations

1. **Indexes Required**:
   - symbol.name (for lookups)
   - call_edge.caller_id, call_edge.callee_id (for traversal)
   - file.path (for citation lookup)

2. **Query Optimization**:
   - Use LIMIT clauses on all queries
   - Leverage existing indexes
   - Connection pooling via asyncpg

3. **Caching Strategy**:
   - Database handles query plan caching
   - Connection pool reuse
   - No application-level caching initially

## Migration Notes

No database schema changes required. This implementation uses existing tables and adds query logic in the application layer.