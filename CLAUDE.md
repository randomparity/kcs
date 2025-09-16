# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kernel Context Server (KCS) - Provides ground-truth Linux kernel analysis
via MCP protocol for AI coding assistants.

## Tech Stack

- **Rust 1.75**: Parser, extractor, graph algorithms (performance-critical)
- **Python 3.11**: MCP server, API layer (FastAPI)
- **PostgreSQL 15+**: Graph storage with pgvector for semantic search
- **Tree-sitter**: Fast structural parsing
- **Clang**: Semantic analysis via compile_commands.json
- **Aya/libbpf-rs**: Optional eBPF tracing

## Project Structure

```text
src/
├── rust/           # Performance-critical Rust components
│   ├── kcs-parser/      # Tree-sitter + clang parsing
│   ├── kcs-extractor/   # Entry point detection
│   ├── kcs-graph/       # Call graph algorithms
│   ├── kcs-impact/      # Impact analysis
│   ├── kcs-drift/       # Drift detection
│   └── kcs-python-bridge/ # Python bindings
├── python/         # MCP server and Python components
│   ├── kcs_mcp/         # FastAPI MCP protocol server
│   ├── kcs_summarizer/  # LLM-powered summaries
│   └── kcs_ci/          # CI utilities
└── sql/           # Database schema and migrations
    ├── migrations/      # Schema versioning
    └── optimizations/   # Performance tuning

tests/
├── contract/      # API contract tests
├── integration/   # Cross-component tests
├── performance/   # Benchmarks
└── fixtures/      # Test data (mini-kernel)

tools/             # Development and deployment scripts
├── index_kernel.sh     # Main kernel indexing script
└── setup/             # Installation helpers
```

## Key Concepts

- **Entry Points**: Kernel boundaries (syscalls, ioctls, file_ops, sysfs, etc.)
- **Call Graph**: Function relationships with config awareness
- **Citations**: Every claim has file:line references
- **Impact Analysis**: Blast radius of changes
- **Drift Detection**: Spec vs implementation mismatches

## Constitutional Requirements

1. **Read-Only**: Never modify kernel source
2. **Citations Required**: All results include file:line spans
3. **MCP-First**: All features via Model Context Protocol
4. **Config-Aware**: Results tagged with kernel configuration
5. **Performance**: Index ≤20min, queries p95 ≤600ms

## MCP API Endpoints

- `search_code`: Semantic/lexical code search
- `get_symbol`: Symbol information with summary
- `who_calls`: Find callers of a function
- `list_dependencies`: Find callees
- `entrypoint_flow`: Trace from entry to implementation
- `impact_of`: Analyze change blast radius
- `diff_spec_vs_code`: Detect drift
- `owners_for`: Find maintainers

## Testing Strategy

- **TDD Required**: Tests before implementation
- **Order**: Contract → Integration → E2E → Unit
- **Real Dependencies**: Use actual Postgres, kernel repos
- **Performance**: k6 for load testing, benchmarks for critical paths

## Development Commands

### Setup and Build

```bash
# Initial setup (creates venv, installs deps, builds Rust)
make setup

# Build Rust components only
make build-rust

# Activate development environment
source .venv/bin/activate
```

### Code Quality

```bash
# Run all quality checks (lint + test)
make check

# Linting only
make lint                # All linting (Python, Rust, YAML, SQL)
make lint-python         # Python only (ruff, mypy)
make lint-rust           # Rust only (clippy)

# Code formatting
make format              # All formatting
make format-python       # Python only
make format-rust         # Rust only
```

### Testing

```bash
# Run all tests
make test

# Specific test types
make test-unit           # Unit tests
make test-integration    # Integration tests
make test-contract       # API contract tests
make test-performance    # Performance benchmarks
```

### Development Server

```bash
# Start MCP server (after building)
kcs-mcp --host 0.0.0.0 --port 8080

# Or via Python module
python -m kcs_mcp.cli --host 0.0.0.0 --port 8080
```

### Docker Operations

```bash
# Start all services via Docker
make docker-compose-up-app        # Core services only
make docker-compose-up-all        # With monitoring

# Check service health
docker compose ps
curl http://localhost:8080/health
```

### Kernel Indexing

```bash
# Index a kernel repository (requires tools built)
tools/index_kernel.sh ~/src/linux

# With specific configuration
tools/index_kernel.sh --config arm64:defconfig ~/src/linux

# Incremental update
tools/index_kernel.sh --incremental ~/src/linux
```

## Database Schema

- **Nodes**: File, Symbol, EntryPoint, KconfigOption
- **Edges**: CallEdge, DependsOn, ModuleSymbol
- **Aggregates**: Summary, DriftReport, TestCoverage
- **Config Bitmap**: Efficient multi-config tagging

## Performance Targets

- Full index: ≤20 minutes
- Incremental: ≤3 minutes
- Query p95: ≤600ms
- Graph size: <20GB for 6 configs
- Scale: ~50k symbols, ~10k entry points

## Build System Architecture

- **Multi-language**: Rust workspace + Python package + SQL migrations
- **Rust workspace**: All Rust crates under `src/rust/` with shared dependencies
- **Python packaging**: Uses setuptools with `pyproject.toml`, installed as editable package
- **Development environment**: Virtual environment via `uv` for fast dependency resolution
- **Code quality**: Enforced via pre-commit hooks (ruff, mypy, clippy, detect-secrets)

## Database Integration

- **PostgreSQL**: Primary storage with pgvector extension for semantic search
- **Migrations**: SQL scripts in `src/sql/migrations/`
- **Connection**: Uses asyncpg for async Python database operations
- **Performance**: Includes optimization scripts in `src/sql/optimizations/`

## Key Dependencies

### Rust Components

- `tree-sitter`/`tree-sitter-c`: Fast syntactic parsing
- `clang-sys`: Semantic analysis integration
- `sqlx`: Type-safe database queries
- `pyo3`: Python bindings for Rust components

### Python Components

- `fastapi`/`uvicorn`: MCP protocol server
- `asyncpg`: PostgreSQL async driver
- `pgvector`: Vector similarity search
- `pydantic`: Data validation and serialization

## Testing Approach

- **Contract-first**: API contracts tested before implementation
- **Real data**: Uses actual kernel repositories, not mocks
- **Performance**: Includes benchmarks with k6 load testing
- **Fixtures**: Mini-kernel test data for fast iteration

---

*When working on KCS:*

1. Always include citations in responses
2. Respect read-only constraint
3. Test with real kernel repositories
4. Monitor performance against targets
5. Use `make check` before committing
6. Activate virtual environment: `source .venv/bin/activate`
