# Phase 0: Research & Discovery

**Feature**: Infrastructure Core Components Implementation
**Date**: 2025-09-17
**Input**: Feature specification analysis and codebase investigation

## Current State Analysis

### 1. Kernel Configuration Parsing

**Current State**:

- Empty stub file at `src/rust/kcs-parser/src/config.rs` with TODO comment
- No kernel config parsing implementation exists
- System hardcoded to x86_64:defconfig assumptions

**Decision**: Implement new `kcs-config` crate
**Rationale**:

- Required for multi-architecture support (ARM64, RISC-V, PowerPC)
- Essential for config-aware analysis per constitution principle IV
- Affects all downstream analysis accuracy

**Alternatives Considered**:

- Using existing Linux kconfig tools: Rejected - requires kernel build environment
- Parsing only .config files: Rejected - misses Kconfig dependency resolution
- Static config database: Rejected - doesn't scale to custom configs

### 2. Drift Detection (Spec vs Implementation)

**Current State**:

- `kcs-drift` crate exists but key modules commented out:
  - `drift_detector` module TODO
  - `report_generator` module TODO
- Basic spec parser implemented but incomplete

**Decision**: Complete `kcs-drift` implementation
**Rationale**:

- Core feature for validation workflows
- Spec parser foundation already exists
- Required for FR-003 and FR-004

**Alternatives Considered**:

- External diff tools: Rejected - need semantic understanding of code
- Manual validation only: Rejected - doesn't scale
- AST-only comparison: Rejected - misses behavioral differences

### 3. Semantic Search

**Current State**:

- Database schema ready with pgvector extension
- `symbol_embedding` table with VECTOR(768) column
- HNSW index configured for vector similarity
- Python MCP has stub `search_code_semantic` function
- No embedding generation or query implementation

**Decision**: Implement `kcs-search` crate with embedding pipeline
**Rationale**:

- pgvector infrastructure already in place
- Enables context-aware search beyond keywords
- Required for FR-005

**Alternatives Considered**:

- Full-text search only: Rejected - misses semantic relationships
- External search service: Rejected - adds complexity and latency
- Pre-computed embeddings only: Rejected - needs dynamic query embedding

### 4. Call Graph Traversal

**Current State**:

- `kcs-graph` crate exists with basic structure
- Path reconstruction marked as TODO
- Cycle detection not implemented
- Graph serialization placeholder

**Decision**: Enhance `kcs-graph` with missing features
**Rationale**:

- Foundation exists, needs completion
- Critical for understanding code relationships
- Required for FR-006 and FR-007

**Alternatives Considered**:

- Graph database (Neo4j): Rejected - adds operational complexity
- In-memory only: Rejected - doesn't scale to full kernel
- Static pre-computation: Rejected - needs dynamic traversal

### 5. Graph Serialization

**Current State**:

- Placeholder comment in `kcs-graph/src/main.rs`
- No export functionality implemented

**Decision**: Implement `kcs-serializer` crate
**Rationale**:

- Clean separation of concerns
- Multiple format support (JSON Graph, GraphML per FR-008)
- Size-aware chunking for large graphs (FR-009)

**Alternatives Considered**:

- Built into kcs-graph: Rejected - violates single responsibility
- Streaming only: Rejected - some tools need complete files
- Binary format: Rejected - not tool-friendly

## Technical Decisions

### Performance Strategy

**Decision**: Streaming + chunking for large data
**Rationale**:

- 50,000+ files on 16GB RAM constraint (FR-009)
- Output must be jq-parseable (not 3GB files)
- Constitutional performance targets

### Integration Points

**Decision**: Maintain clear crate boundaries with CLI interfaces
**Rationale**:

- Each crate independently testable
- Follows existing KCS architecture
- Enables parallel development

### Database Migrations

**Decision**: Incremental migrations for new features
**Rationale**:

- pgvector already configured
- Preserve existing data
- Backward compatibility

## Risk Analysis

### High Risk

- **Embedding model selection**: Wrong choice affects search quality
- **Graph memory usage**: Full kernel graph may exceed RAM

### Medium Risk

- **Config format variations**: Different kernel versions have different Kconfig
- **Spec format parsing**: Unstructured docs are hard to validate

### Low Risk

- **Database schema changes**: Well-understood domain
- **CLI interfaces**: Following established patterns

## Dependencies Validation

All primary dependencies verified as already in use:

- tree-sitter: ✓ Used in kcs-parser
- clang-sys: ✓ Used for semantic enhancement
- PyO3: ✓ Python bridge exists
- pgvector: ✓ Database configured
- FastAPI: ✓ MCP server running

No new external dependencies required.

## Next Steps (Phase 1)

1. Define data models for:
   - Kernel configuration representation
   - Drift detection results
   - Semantic search queries/results
   - Graph serialization formats

2. Design MCP contract extensions for new capabilities

3. Create failing tests for each new component

## Conclusion

All technical unknowns resolved. Existing infrastructure supports planned implementation. No
blocking dependencies or architectural changes required.
