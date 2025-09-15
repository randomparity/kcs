# Research Findings: Kernel Context Server

**Date**: 2025-09-14
**Feature**: KCS - Kernel Context Server

## Technical Decisions

### 1. Tree-sitter Grammar for Kernel C
**Decision**: Use standard tree-sitter-c with custom queries for kernel patterns
**Rationale**:
- Standard grammar handles 95% of kernel code correctly
- Custom queries can identify kernel-specific patterns (EXPORT_SYMBOL, module_param, etc.)
- Faster than maintaining a fork of the grammar
**Alternatives considered**:
- Fork tree-sitter-c: Too much maintenance overhead
- Sparse/Coccinelle: Good for semantic patches but not general parsing
- Pure clang AST: Too slow for full repository scanning

### 2. Clang Index Integration
**Decision**: Generate compile_commands.json per configuration using kernel's make system
**Rationale**:
- `make compile_commands.json` native kernel support since v5.10
- Provides accurate preprocessor state for each config
- Handles complex macro expansions correctly
**Alternatives considered**:
- Bear tool: Additional dependency, less accurate
- Manual generation: Error-prone, maintenance burden
- cscope/ctags: Insufficient for modern kernel analysis

### 3. pgvector Indexing Strategy
**Decision**: Hybrid approach - exact match for symbols, vector search for semantic queries
**Rationale**:
- Symbol lookups need exact matching (hash index)
- Code search benefits from semantic similarity (HNSW index)
- Separate tables for different query patterns
**Alternatives considered**:
- Pure vector search: Too slow for exact lookups
- Pure text search: Misses semantic relationships
- Elasticsearch: Overkill for our use case

### 4. MCP Protocol Implementation
**Decision**: FastAPI with async handlers, JSON-RPC 2.0 transport
**Rationale**:
- MCP spec recommends JSON-RPC 2.0
- FastAPI provides automatic OpenAPI generation
- Async handlers crucial for concurrent queries
**Alternatives considered**:
- gRPC: Not MCP-compliant
- Raw ASGI: Too low-level
- Django: Too heavyweight for our needs

### 5. eBPF Tracing Integration
**Decision**: Aya framework for Rust-native eBPF
**Rationale**:
- Pure Rust, better integration with parser components
- Compile-time safety for eBPF programs
- Active development and good documentation
**Alternatives considered**:
- libbpf-rs: C bindings add complexity
- bcc: Python-based, performance overhead
- Manual BPF: Too error-prone

### 6. Multi-Config Edge Tagging
**Decision**: Bitmap representation for config presence
**Rationale**:
- Efficient storage (8 configs = 1 byte per edge)
- Fast bitwise operations for filtering
- Scales to dozens of configs if needed
**Alternatives considered**:
- Separate tables per config: Too much duplication
- JSON arrays: Inefficient queries
- Normalized junction table: Too many joins

## Architecture Patterns

### Parser Pipeline
```
Kernel Repo → Tree-sitter (structure) → Clang (semantics) → Graph Builder → Postgres
                     ↓                         ↓                    ↓
              Custom Queries            Macro Resolution      Edge Detection
```

### Query Flow
```
MCP Request → FastAPI → Query Planner → Postgres → Citation Formatter → MCP Response
                 ↓            ↓              ↓              ↓
            Auth Check    Cache Check    pgvector      Line Mapping
```

### Impact Analysis Algorithm
```
1. Parse diff → extract changed symbols
2. Graph traversal with config-aware edges
3. Prune at module boundaries (unless exported)
4. Collect affected: configs, tests, owners
5. Risk scoring based on context (irq, locks, etc.)
```

## Performance Optimizations

### Indexing
- Parallel parsing with rayon (Rust)
- Incremental updates via file mtimes
- Bloom filters for symbol existence checks
- Connection pooling for Postgres writes

### Querying
- Redis cache for frequent queries
- Prepared statements for common patterns
- Lazy loading of summaries
- Stream results for large datasets

### Storage
- TOAST compression for large text fields
- Partitioning by configuration
- Periodic VACUUM ANALYZE
- Index-only scans where possible

## Security Considerations

### Authentication
- JWT tokens with per-project claims
- Rate limiting per token
- Audit log for all queries
- No direct SQL access

### Data Protection
- Source code never logged
- PII redaction in summaries
- Encrypted connections only
- Read-only database user for queries

## Integration Points

### CI Systems
- GitHub Actions: Native webhook support
- Jenkins: REST API adapter
- GitLab CI: Direct integration
- Generic: JSON artifact format

### Development Tools
- VS Code: MCP client extension
- Neovim: LSP bridge
- Emacs: eglot integration
- Claude/Copilot: Native MCP support

## Validation Strategy

### Correctness
- Known pattern test suite (syscalls, ioctls, etc.)
- Comparison with cscope/ctags output
- Manual verification of complex cases
- Kernel maintainer feedback

### Performance
- k6 load tests for API endpoints
- Benchmark suite for indexing
- Query performance regression tests
- Memory profiling with valgrind/heaptrack

### Coverage
- Entry point detection rate
- Edge resolution accuracy
- Symbol coverage percentage
- Config-specific validation

## Risk Mitigation

### Technical Risks
- **Macro complexity**: Fallback to clang when tree-sitter fails
- **Scale issues**: Horizontal sharding if needed
- **Version skew**: Pin parser versions, test on LTS kernels
- **Dynamic behavior**: Optional tracing to confirm static analysis

### Operational Risks
- **Database growth**: Retention policies, archival strategy
- **Query storms**: Circuit breakers, backpressure
- **Stale indices**: Automatic reindexing triggers
- **Config explosion**: Limit to key configurations initially

## Dependencies and Licenses

### Rust Dependencies (MIT/Apache-2.0 compatible)
- tree-sitter: MIT
- clang-sys: Apache-2.0
- tokio: MIT
- pyo3: Apache-2.0
- aya: Apache-2.0/MIT

### Python Dependencies (BSD/MIT compatible)
- fastapi: MIT
- asyncpg: Apache-2.0
- structlog: Apache-2.0
- pydantic: MIT

### Infrastructure (Open Source)
- PostgreSQL: PostgreSQL License
- pgvector: PostgreSQL License
- Redis: BSD

## Next Steps

1. Create detailed data model (Phase 1)
2. Define MCP contract specifications
3. Set up development environment
4. Create failing contract tests
5. Begin incremental implementation

---

*All technical decisions validated against constitutional requirements for performance, security, and simplicity.*