# CLAUDE.md - KCS Project Context

## Project Overview

Kernel Context Server (KCS) - Provides ground-truth Linux kernel analysis via MCP protocol for AI coding assistants.

## Tech Stack

- **Rust 1.75**: Parser, extractor, graph algorithms (performance-critical)
- **Python 3.11**: MCP server, API layer (FastAPI)
- **PostgreSQL 15+**: Graph storage with pgvector for semantic search
- **Tree-sitter**: Fast structural parsing
- **Clang**: Semantic analysis via compile_commands.json
- **Aya/libbpf-rs**: Optional eBPF tracing

## Project Structure

```
src/
├── rust/           # Performance-critical components
│   ├── kcs-parser/     # Tree-sitter + clang
│   ├── kcs-extractor/  # Entry point detection
│   ├── kcs-graph/      # Graph algorithms
│   ├── kcs-impact/     # Impact analysis
│   └── kcs-drift/      # Drift detection
├── python/         # API and integration
│   ├── kcs_mcp/        # MCP protocol server
│   └── kcs_summarizer/ # LLM summaries
└── sql/           # Database schema

tests/
├── contract/      # API contract tests
├── integration/   # Cross-component tests
└── performance/   # Benchmarks
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

## Common Commands

```bash
# Parse kernel
kcs-parser --parse ~/linux --config x86_64:defconfig

# Extract entry points
kcs-extractor --extract syscalls --input index.json

# Query graph
kcs-graph --query who_calls --symbol vfs_read

# Start MCP server
kcs-mcp --serve --port 8080 --auth-token TOKEN

# Run impact analysis
kcs-impact --diff changes.patch --depth 2

# Check drift
kcs-drift --spec feature.yaml --code ~/linux
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

## Recent Changes

- Initial project setup and architecture design
- MCP API contract definition
- Database schema with pgvector integration

---

*When working on KCS:*

1. Always include citations in responses
2. Respect read-only constraint
3. Test with real kernel repositories
4. Monitor performance against targets
