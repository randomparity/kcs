# Research: Implement MCP Tools

**Date**: 2025-09-16
**Feature**: Replace mock implementations with actual database-backed functionality

## Overview

This research document consolidates findings for implementing the actual MCP tool endpoints to
replace the current mock implementations. All technical context is already established in the
existing KCS project.

## Key Decisions

### 1. Database Query Strategy

**Decision**: Use existing asyncpg database methods with recursive CTEs for graph traversal
**Rationale**:

- Already implemented find_callers() and find_callees() methods in database.py
- PostgreSQL CTEs handle depth-limited traversal efficiently
- Avoids loading entire graph into memory
**Alternatives considered**:
- GraphQL resolver pattern: Too complex for current needs
- In-memory graph library: Would violate performance constraints at scale
- Neo4j migration: Unnecessary complexity, PostgreSQL sufficient

### 2. Circular Dependency Handling

**Decision**: Use visited set tracking with depth limits
**Rationale**:

- Simple and proven approach for cycle detection
- Already partially implemented in entrypoint_flow endpoint
- Depth limits provide natural termination
**Alternatives considered**:
- Tarjan's algorithm: Overkill for simple traversal needs
- Database-level cycle detection: Complex SQL, harder to debug

### 3. Citation Management

**Decision**: Leverage existing Span objects from database results
**Rationale**:

- Database already returns span information (path, sha, start, end)
- Consistent with constitutional requirement for citations
- Span model already defined in MCP protocol
**Alternatives considered**:
- Separate citation service: Unnecessary abstraction
- File-based lookups: Would violate performance requirements

### 4. Entry Point Mapping

**Decision**: Maintain syscall number to function mapping dictionary
**Rationale**:

- Simple, fast lookups for common entry points
- Already partially implemented in entrypoint_flow
- Can be extended with ioctl, file_ops mappings later
**Alternatives considered**:
- Database table for mappings: Overhead for static data
- Parse from kernel headers: Complex, version-dependent

### 5. Impact Analysis Algorithm

**Decision**: Bidirectional traversal with subsystem pattern matching
**Rationale**:

- Find both callers (affected by changes) and callees (dependencies)
- Use symbol name patterns to identify subsystems (vfs_, ext4_, net_)
- Aggregate risk based on blast radius size
**Alternatives considered**:
- ML-based impact prediction: No training data available
- Static analysis only: Misses runtime dependencies
- Manual maintainer mapping: High maintenance burden

### 6. Performance Optimization

**Decision**: Apply result limits and connection pooling
**Rationale**:

- Constitutional requirement: p95 < 600ms
- Database already uses connection pooling via asyncpg
- LIMIT clauses prevent runaway queries
**Alternatives considered**:
- Redis caching layer: Complexity for marginal gains
- Materialized views: Cache invalidation complexity
- Query parallelization: Database handles this internally

### 7. Error Handling

**Decision**: Graceful degradation with empty results on failures
**Rationale**:

- Better UX than errors for missing data
- Allows partial functionality during index updates
- Consistent with existing mock fallback pattern
**Alternatives considered**:
- Fail fast on any error: Poor user experience
- Detailed error types: Over-engineering for current scope

### 8. Configuration Filtering

**Decision**: Optional config parameter passed to database methods
**Rationale**:

- Database schema already has config_bitmap support
- Optional parameter maintains backward compatibility
- Can filter at query time efficiently
**Alternatives considered**:
- Separate endpoints per config: API explosion
- Client-side filtering: Inefficient data transfer
- Required config parameter: Breaks existing tests

## Integration Points

### Existing Components to Leverage

1. **Database.find_callers()**: Already queries call_edge table with joins
2. **Database.find_callees()**: Mirror of find_callers for dependencies
3. **Span model**: Existing pydantic model for citations
4. **CallerInfo model**: Reusable for both callers and callees
5. **Error handling**: Existing HTTPException patterns
6. **Logging**: structlog already configured

### Required Modifications

1. Remove mock data fallbacks after implementation
2. Add depth traversal for multi-hop queries
3. Implement visited tracking for cycle prevention
4. Extend entry point mappings beyond syscalls
5. Add subsystem detection logic for impact analysis

## Testing Strategy

### Contract Tests

- Verify response schemas match OpenAPI spec
- Test with empty database (graceful handling)
- Test with populated database (actual results)

### Integration Tests

- Test against real PostgreSQL with testcontainers
- Seed with known call graph data
- Verify depth limiting works correctly
- Test circular dependency handling

### Performance Tests

- Measure query times with production-scale data
- Verify p95 < 600ms constitutional requirement
- Test connection pool behavior under load

## Risk Mitigation

### Performance Risks

- **Risk**: Deep traversals could timeout
- **Mitigation**: Hard depth limits (max 5 levels)

### Data Quality Risks

- **Risk**: Incomplete call graphs from parser
- **Mitigation**: Graceful handling of missing edges

### Backward Compatibility

- **Risk**: Breaking existing test suites
- **Mitigation**: Keep mock fallbacks initially, remove after validation

## Next Steps

1. Generate API contracts from existing endpoint signatures
2. Create failing contract tests
3. Implement database traversal logic
4. Remove mock implementations
5. Validate performance requirements

## Conclusions

The implementation path is clear with existing infrastructure in place. The primary work involves:

1. Enhancing database methods for depth traversal
2. Adding cycle detection
3. Implementing subsystem pattern matching
4. Removing mock data
5. Ensuring all results include proper citations

No blocking technical decisions remain. The existing codebase provides all necessary foundations.
