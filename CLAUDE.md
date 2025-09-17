# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kernel Context Server (KCS) - Provides ground-truth Linux kernel analysis via MCP protocol for AI coding assistants.

## Architecture Overview

### Multi-Language Pipeline

KCS uses a three-stage pipeline architecture:

1. **Rust Components** (Performance-critical parsing/analysis)
   - `kcs-parser`: Tree-sitter + Clang for syntactic/semantic parsing
   - `kcs-extractor`: Entry point detection (syscalls, ioctls, file_ops, sysfs)
   - `kcs-graph`: Call graph algorithms with cycle detection
   - `kcs-impact`: Change impact analysis
   - `kcs-drift`: Spec vs implementation comparison
   - `kcs-config`: Kernel configuration parsing (planned)
   - `kcs-search`: Semantic search with pgvector (planned)
   - `kcs-serializer`: Graph export to JSON/GraphML (planned)
   - `kcs-python-bridge`: PyO3 bindings for Python integration

2. **Python MCP Server** (API and orchestration)
   - FastAPI server exposing MCP protocol endpoints
   - Async PostgreSQL queries via asyncpg
   - Connection pooling with 2-10 connections
   - Structured logging with structlog

3. **PostgreSQL Database** (Graph storage)
   - pgvector extension for semantic search
   - JSONB metadata columns for extensibility
   - Recursive CTEs for graph traversal with depth limiting

### Data Flow

```
Kernel Source → Rust Parser → Entry Points/Symbols → PostgreSQL
                     ↓
              Clang Enhancement
                     ↓
              Python Bridge → MCP Server → AI Assistant
```

### Key Design Patterns

- **Streaming Processing**: Tools output newline-delimited JSON for memory efficiency
- **Progressive Enhancement**: Basic parsing works without Clang, enriched when available
- **Configuration Awareness**: All data tagged with kernel config (x86_64:defconfig, etc.)
- **Citation-First**: Every result includes file:line:sha references

## Development Commands

### Quick Start

```bash
# One-time setup (uses uv for fast Python dependency management)
make setup              # Creates venv, installs deps, builds Rust, installs hooks

# Activate environment (required for all Python commands)
source .venv/bin/activate

# Development cycle
make check              # Runs lint + test (use before committing)
make format             # Auto-format all code
make ci                 # Run full CI pipeline locally
```

### Building Components

```bash
# Build all Rust crates
cd src/rust && cargo build --release --workspace

# Build specific crate
cd src/rust/kcs-parser && cargo build --release

# Install Python package in development mode (after activating venv)
pip install -e ".[dev,performance]"
```

### Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-unit          # Fast unit tests
make test-integration   # Integration tests (requires DB)
make test-contract      # API contract validation
make test-performance   # Performance benchmarks

# Run single Python test
pytest tests/contract/test_mcp_who_calls.py::test_who_calls_basic -v

# Run single Rust test
cd src/rust/kcs-parser && cargo test test_parse_function -- --nocapture

# Run tests with coverage
pytest --cov=src/python/kcs_mcp --cov-report=html tests/
```

### Code Quality

```bash
# Python linting
ruff check src/python/ tests/ tools/    # Fast Python linter
mypy src/python/                        # Type checking

# Rust linting
cd src/rust && cargo clippy --all-targets --all-features -- -D warnings

# Format code
ruff format src/python/ tests/          # Python formatting
cd src/rust && cargo fmt --all          # Rust formatting

# SQL linting
sqlfluff lint src/sql/                  # SQL style checking
```

### Running the Server

```bash
# Start MCP server locally (requires activated venv)
kcs-mcp --host 0.0.0.0 --port 8080

# With custom database URL
DATABASE_URL=postgresql://user:pass@localhost/kcs kcs-mcp

# Docker compose (full stack)
make docker-compose-up-app              # Core services only
make docker-compose-up-all              # With monitoring (Grafana, Prometheus)

# Check health
curl http://localhost:8080/health
```

### Indexing Kernels

```bash
# Full kernel indexing
tools/index_kernel.sh ~/src/linux

# Subsystem only (faster for testing)
tools/index_kernel.sh --subsystem fs/ext4 ~/src/linux

# With specific config
tools/index_kernel.sh --config arm64:defconfig ~/src/linux

# Incremental update
tools/index_kernel.sh --incremental ~/src/linux

# Extract entry points only (streaming)
tools/extract_entry_points_streaming.py ~/src/linux | head -20
```

### Database Operations

```bash
# Connect to database
psql -d kcs -U kcs -h localhost

# Apply migrations
for f in src/sql/migrations/*.sql; do psql -d kcs -f "$f"; done

# Common queries for debugging
psql -d kcs -c "SELECT COUNT(*) FROM entry_point GROUP BY entry_type;"
psql -d kcs -c "SELECT name, file_path FROM symbol WHERE name LIKE 'vfs_%' LIMIT 10;"

# Check recursive CTE performance
psql -d kcs -c "EXPLAIN ANALYZE WITH RECURSIVE ..."
```

## Project Structure

```text
src/
├── rust/               # Performance-critical Rust components
│   ├── kcs-*/             # Individual crates with Cargo.toml
│   └── Cargo.toml         # Workspace configuration
├── python/             # MCP server and Python components
│   └── kcs_mcp/           # FastAPI application
│       ├── __init__.py
│       ├── cli.py         # Entry point
│       ├── database.py    # Connection pooling and queries
│       ├── models.py      # Pydantic models
│       ├── tools.py       # MCP endpoint implementations
│       └── server.py      # FastAPI app setup
└── sql/                # Database schema
    └── migrations/        # Numbered migration files

tests/
├── contract/           # API contract tests (OpenAPI validation)
├── integration/        # Cross-component tests
├── performance/        # k6 and benchmark tests
└── fixtures/          # Test kernel code samples

tools/                  # Shell and Python scripts
├── index_kernel.sh       # Main indexing orchestrator
└── extract_entry_points_streaming.py  # Streaming parser wrapper
```

## Constitutional Requirements

1. **Read-Only**: Never modify kernel source - all operations are analysis only
2. **Citations Required**: All results include `Span(file, line, sha)` objects
3. **MCP-First**: All features exposed via `/mcp/tools/*` endpoints
4. **Config-Aware**: Results tagged with kernel configuration context
5. **Performance**: Index ≤20min, queries p95 ≤600ms (enforced by tests)

## Key Implementation Details

### Database Schema

- **Recursive CTEs**: Used for call graph traversal with cycle detection

  ```sql
  WITH RECURSIVE callers_tree AS (
    -- Base case
    SELECT caller_id, 1 as depth, ARRAY[caller_id] as visited
    -- Recursive case with cycle check
    WHERE depth < $2 AND NOT (caller_id = ANY(visited))
  )
  ```

- **JSONB Metadata**: Extensible without schema changes

  ```sql
  metadata->>'export_type' = 'GPL'  -- Indexed for performance
  ```

### Entry Point Detection

Current implementation in `src/rust/kcs-extractor/src/entry_points.rs`:

- Syscalls: Mapped from `__NR_*` to `sys_*` functions
- File ops: Regex patterns for `file_operations` structs
- Sysfs: `DEVICE_ATTR` macro patterns
- Ioctls: Partially implemented, needs magic number extraction

### MCP Endpoint Pattern

All endpoints in `src/python/kcs_mcp/tools.py` follow:

1. Validate request with Pydantic model
2. Build SQL query with depth/config parameters
3. Execute with connection pool
4. Include citations in response
5. Handle empty results gracefully

### Performance Optimization

- **Parallel Processing**: Rust crates use rayon for file-level parallelism
- **Streaming**: Python tools yield results as they're produced
- **Connection Pooling**: 2-10 PostgreSQL connections
- **Query Limits**: LIMIT 100 on recursive queries to prevent explosion
- **Depth Limits**: Maximum depth 5 for graph traversal

## Testing Approach

- **TDD Required**: Tests must fail before implementation (RED-GREEN-Refactor)
- **Contract Tests**: Validate OpenAPI schemas before implementation
- **Real Data**: Use actual kernel code in `tests/fixtures/kernel/`
- **Performance Tests**: `tests/performance/test_mcp_performance.py` validates p95 targets

## Current Development Status

### Infrastructure Components (Branch: 005-infrastructure-empty-stub)
- **kcs-config**: Empty stub - kernel config parsing not implemented
- **kcs-drift**: Module exists but `drift_detector` and `report_generator` commented out
- **kcs-search**: Semantic search DB schema ready, implementation pending
- **kcs-graph**: Path reconstruction and cycle detection marked as TODO
- **kcs-serializer**: Graph export placeholder only

### Working Features
- Basic tree-sitter parsing with call graph extraction (use `--include-calls` flag)
- Entry point detection for syscalls and file_ops
- PostgreSQL with pgvector configured
- MCP endpoints for basic queries

## Common Pitfalls

1. **Missing Virtual Environment**: Always `source .venv/bin/activate` before Python work
2. **Rust Not Built**: Run `make build-rust` after Rust changes
3. **Database Not Running**: Use `docker compose up postgres` or local PostgreSQL
4. **Migrations Not Applied**: Check `src/sql/migrations/` are applied
5. **Clang Not Found**: Clang integration gracefully degrades if unavailable
6. **Call Graph Extraction**: Must use `--include-calls` flag with kcs-parser for relationships

---

*When working on KCS:*

1. Always include citations in responses (file:line references)
2. Respect read-only constraint on kernel source
3. Test with real kernel repositories when possible
4. Monitor performance against constitutional targets
5. Use `make check` before committing
6. Activate virtual environment: `source .venv/bin/activate`
