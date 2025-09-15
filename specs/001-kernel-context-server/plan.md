# Implementation Plan: Kernel Context Server (KCS)

**Branch**: `001-kernel-context-server` | **Date**: 2025-09-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-kernel-context-server/spec.md`

## Execution Flow (/plan command scope)

```
1. Load feature spec from Input path
   → Feature spec loaded successfully
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Project type: System service (kernel analysis + MCP server)
   → Structure Decision: Option 1 (single project)
3. Evaluate Constitution Check section below
   → No violations identified
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → Research completed with architecture decisions
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
6. Re-evaluate Constitution Check section
   → No new violations
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:

- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

The Kernel Context Server (KCS) provides a ground-truth, queryable model of the Linux kernel for AI coding tools and engineers. It parses kernel repositories into graphs, extracts entry points and dependencies, maintains summaries, and exposes everything via Model Context Protocol (MCP). Technical approach: Rust extractors for performance-critical parsing, Python MCP server for API flexibility, Postgres with pgvector for graph storage and semantic search, and optional eBPF tracing via Aya/libbpf-rs.

## Technical Context

**Language/Version**: Rust 1.75 (extractors), Python 3.11 (MCP server)
**Primary Dependencies**: tree-sitter, clang-sys, tokio, pyo3, fastapi, asyncpg, pgvector, aya/libbpf-rs
**Storage**: PostgreSQL 15+ with pgvector extension
**Testing**: cargo test (Rust), pytest (Python), k6 (performance)
**Target Platform**: Linux x86_64 (primary), ppc64le, s390x (multi-arch support)
**Project Type**: single - System service with multiple components
**Performance Goals**: Full index ≤20min, incremental ≤3min, query p95 ≤600ms
**Constraints**: Read-only operations, <20GB graph storage, token-based auth required
**Scale/Scope**: ~50k symbols, ~10k entry points, 3 architectures × 2 configs each

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 1 (single monorepo with Rust + Python components)
- Using framework directly? Yes (FastAPI for MCP, tokio for async)
- Single data model? Yes (shared Postgres schema)
- Avoiding patterns? Yes (no unnecessary abstractions)

**Architecture**:

- EVERY feature as library? Yes
- Libraries listed:
  - kcs-parser: Rust library for kernel code parsing
  - kcs-extractor: Rust library for entry point extraction
  - kcs-graph: Rust library for graph operations
  - kcs-mcp: Python library for MCP protocol implementation
  - kcs-impact: Rust library for impact analysis
  - kcs-drift: Rust library for drift detection
- CLI per library:
  - kcs-parser --parse <repo> --config <config> --format json
  - kcs-extractor --extract <type> --input <index> --format json
  - kcs-graph --query <operation> --symbol <name> --format json
  - kcs-mcp --serve --port 8080 --auth-token <token>
  - kcs-impact --diff <file> --depth <n> --format json
  - kcs-drift --spec <file> --code <path> --format json
- Library docs: llms.txt format planned? Yes

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? Yes
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (real Postgres, real kernel repos)
- Integration tests for: new libraries, contract changes, shared schemas? Yes
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:

- Structured logging included? Yes (tracing for Rust, structlog for Python)
- Frontend logs → backend? N/A (no frontend)
- Error context sufficient? Yes (spans with file/line citations)

**Versioning**:

- Version number assigned? 1.0.0
- BUILD increments on every change? Yes
- Breaking changes handled? Yes (MCP protocol versioning)

## Project Structure

### Documentation (this feature)

```
specs/001-kernel-context-server/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

```
# Option 1: Single project (SELECTED)
src/
├── rust/
│   ├── kcs-parser/      # Tree-sitter + clang parsing
│   ├── kcs-extractor/   # Entry point extraction
│   ├── kcs-graph/       # Graph algorithms
│   ├── kcs-impact/      # Impact analysis
│   ├── kcs-drift/       # Drift detection
│   └── kcs-tracer/      # Optional eBPF tracing
├── python/
│   ├── kcs_mcp/         # MCP server implementation
│   ├── kcs_summarizer/  # LLM-assisted summaries
│   └── kcs_ci/          # CI integration adapter
└── sql/
    └── migrations/      # Database schema migrations

tests/
├── contract/            # MCP API contract tests
├── integration/         # Cross-component tests
├── performance/         # Latency/throughput benchmarks
└── fixtures/            # Sample kernel repos

tools/
├── setup/               # Installation scripts
└── ci/                  # CI integration scripts
```

**Structure Decision**: Option 1 (single project with multi-language components)

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - Optimal tree-sitter grammar for kernel C
   - Clang index integration approach
   - pgvector indexing strategy for code search
   - MCP protocol implementation details
   - eBPF tracing integration with Aya
   - Multi-config edge tagging approach

2. **Generate and dispatch research agents**:

   ```
   Task: "Research tree-sitter C grammar customization for kernel macros"
   Task: "Find best practices for clang compile_commands.json generation"
   Task: "Research pgvector performance for 50k+ node graphs"
   Task: "Analyze MCP protocol specification and reference implementations"
   Task: "Evaluate Aya vs libbpf-rs for kernel tracing"
   Task: "Research multi-config build matrix strategies"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical decisions resolved

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - File, Symbol, EntryPoint, KconfigOption entities
   - CallEdge, DependsOn, ModuleSymbol relationships
   - Summary, Citation, DriftReport aggregates

2. **Generate API contracts** from functional requirements:
   - MCP resources (read-only): repo://, graph://, kb://, docs://, tests://, owners://
   - MCP tools: search_code, get_symbol, who_calls, impact_of, etc.
   - Output OpenAPI schema to `/contracts/mcp-api.yaml`

3. **Generate contract tests** from contracts:
   - One test per MCP tool/resource
   - Schema validation for all responses
   - Citation format verification

4. **Extract test scenarios** from user stories:
   - Onboarding flow test
   - Impact analysis test
   - Drift detection test
   - CI integration test

5. **Update agent file incrementally**:
   - Run update script for CLAUDE.md
   - Add Rust/Python/Postgres context
   - Include MCP protocol details

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

- Database schema creation tasks [P]
- Rust library scaffolding tasks [P]
- Python MCP server setup tasks
- Parser implementation tasks
- Extractor implementation tasks
- Graph algorithm tasks
- MCP endpoint implementation tasks
- Integration test tasks
- Performance benchmark tasks

**Ordering Strategy**:

- TDD order: Tests before implementation
- Dependency order: Schema → Libraries → Server → Integration
- Mark [P] for parallel execution where possible

**Estimated Output**: 30-35 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

No violations - proceeding with standard approach.

## Progress Tracking

*This checklist is updated during execution flow*

**Phase Status**:

- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.0.0 - See `/constitution.md`*
