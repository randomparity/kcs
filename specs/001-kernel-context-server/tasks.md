# Tasks: Kernel Context Server (KCS)

**Input**: Design documents from `/specs/001-kernel-context-server/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```text
1. Load plan.md from feature directory
   → Tech stack: Rust 1.75, Python 3.11, PostgreSQL 15+
   → Libraries: kcs-parser, kcs-extractor, kcs-graph, kcs-mcp, kcs-impact, kcs-drift
2. Load optional design documents:
   → data-model.md: 10 entities (File, Symbol, EntryPoint, etc.)
   → contracts/mcp-api.yaml: 11 MCP tools + resources
   → research.md: Technical decisions loaded
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests for all MCP endpoints
   → Core: Rust libraries, Python server, DB schema
   → Integration: Cross-component, CI adapter
   → Polish: performance benchmarks, documentation
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001-T040)
6. Return: SUCCESS (tasks ready for execution)
```text

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup & Infrastructure

- [x] T001 Create project structure per implementation plan (src/rust/, src/python/, src/sql/, tests/, tools/)
- [x] T002 Initialize Rust workspace with Cargo.toml for 6 libraries
- [x] T003 Initialize Python project with pyproject.toml for MCP server
- [x] T004 [P] Create Docker Compose configuration for PostgreSQL + pgvector
- [x] T005 [P] Configure GitHub Actions CI workflow in .github/workflows/ci.yml
- [x] T006 [P] Setup pre-commit hooks for Rust (rustfmt, clippy) and Python (black, ruff)

## Phase 3.2: Database Schema (Foundation)

- [x] T007 Create initial migration in src/sql/migrations/001_initial_schema.sql with File, Symbol tables
- [x] T008 Add EntryPoint, CallEdge, KconfigOption tables in src/sql/migrations/002_graph_tables.sql
- [x] T009 Add Summary, DriftReport, TestCoverage tables in src/sql/migrations/003_aggregate_tables.sql
- [x] T010 Create indexes and constraints in src/sql/migrations/004_indexes.sql
- [x] T011 [P] Write migration runner script in tools/setup/migrate.sh

## Phase 3.3: Contract Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.4

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [x] T012 [P] Contract test for search_code tool in tests/contract/test_search_code.py
- [x] T013 [P] Contract test for get_symbol tool in tests/contract/test_get_symbol.py
- [x] T014 [P] Contract test for who_calls tool in tests/contract/test_who_calls.py
- [x] T015 [P] Contract test for list_dependencies tool in tests/contract/test_list_dependencies.py
- [x] T016 [P] Contract test for entrypoint_flow tool in tests/contract/test_entrypoint_flow.py
- [x] T017 [P] Contract test for impact_of tool in tests/contract/test_impact_of.py
- [x] T018 [P] Contract test for diff_spec_vs_code tool in tests/contract/test_diff_spec_vs_code.py
- [x] T019 [P] Contract test for owners_for tool in tests/contract/test_owners_for.py
- [x] T020 [P] Integration test for onboarding flow in tests/integration/test_onboarding_flow.py
- [x] T021 [P] Integration test for impact analysis flow in tests/integration/test_impact_analysis.py
- [x] T022 [P] Integration test for drift detection flow in tests/integration/test_drift_detection.py

## Phase 3.4: Core Rust Libraries (ONLY after tests are failing)

- [x] T023 [P] Scaffold kcs-parser library in src/rust/kcs-parser/ with tree-sitter setup
- [x] T024 [P] Scaffold kcs-extractor library in src/rust/kcs-extractor/ for entry point detection
- [x] T025 [P] Scaffold kcs-graph library in src/rust/kcs-graph/ for graph algorithms
- [x] T026 [P] Scaffold kcs-impact library in src/rust/kcs-impact/ for impact analysis
- [x] T027 [P] Scaffold kcs-drift library in src/rust/kcs-drift/ for drift detection
- [x] T028 Implement parser CLI in src/rust/kcs-parser/src/main.rs with --parse command
- [x] T029 Implement syscall extractor in src/rust/kcs-extractor/src/syscalls.rs
- [x] T030 Implement ioctl decoder in src/rust/kcs-extractor/src/ioctls.rs
- [x] T031 Implement call graph builder in src/rust/kcs-graph/src/builder.rs
- [x] T032 Implement who_calls query in src/rust/kcs-graph/src/queries.rs
- [x] T033 Implement impact analyzer in src/rust/kcs-impact/src/analyzer.rs

## Phase 3.5: Python MCP Server

- [x] T034 Create FastAPI app structure in src/python/kcs_mcp/app.py with JWT auth
- [x] T035 Implement MCP resource endpoints in src/python/kcs_mcp/resources.py
- [x] T036 Implement MCP tool endpoints in src/python/kcs_mcp/tools.py
- [x] T037 Create database connection pool in src/python/kcs_mcp/database.py
- [x] T038 Implement citation formatter in src/python/kcs_mcp/citations.py

## Phase 3.6: Integration & Cross-Component

- [x] T039 Create Python-Rust bridge using PyO3 in src/python/kcs_mcp/rust_bridge.py
- [x] T040 Implement CI adapter in src/python/kcs_ci/adapter.py for GitHub Actions
- [x] T041 Create installation script in tools/setup/install.sh
- [x] T042 Write kernel indexing pipeline script in tools/index_kernel.sh

## Phase 3.7: Performance & Polish

- [x] T043 [P] Create k6 performance test for MCP endpoints in tests/performance/mcp_load.js
- [x] T044 [P] Add benchmark suite for parser in src/rust/kcs-parser/benches/
- [x] T045 [P] Write unit tests for citation formatter in tests/unit/test_citations.py
- [x] T046 [P] Update quickstart validation in tests/integration/test_quickstart.py
- [x] T047 Generate API documentation from OpenAPI spec
- [x] T048 Create deployment guide in docs/deployment.md
- [x] T049 Run full system test with sample kernel repository
- [x] T050 Performance optimization based on profiling results

## Dependencies

- Database schema (T007-T010) blocks everything else
- Contract tests (T012-T022) must fail before implementation (T023-T038)
- Rust libraries (T023-T033) before Python bridge (T039)
- MCP server (T034-T038) requires database (T007-T010)
- Integration (T039-T042) requires both Rust and Python components
- Performance tests (T043-T044) after implementation complete

## Parallel Execution Examples

```bash
# Launch all contract tests together (T012-T019):
Task agent="test-writer" prompt="Create contract test for search_code MCP tool"
Task agent="test-writer" prompt="Create contract test for get_symbol MCP tool"
Task agent="test-writer" prompt="Create contract test for who_calls MCP tool"
Task agent="test-writer" prompt="Create contract test for impact_of MCP tool"

# Launch all Rust library scaffolding (T023-T027):
Task agent="rust-dev" prompt="Scaffold kcs-parser library with tree-sitter"
Task agent="rust-dev" prompt="Scaffold kcs-extractor library"
Task agent="rust-dev" prompt="Scaffold kcs-graph library"
Task agent="rust-dev" prompt="Scaffold kcs-impact library"
Task agent="rust-dev" prompt="Scaffold kcs-drift library"
```text

## Critical Path

1. **Database first**: T007-T010 (sequential, ~2 hours)
2. **Tests next**: T012-T022 (parallel, ~3 hours)
3. **Core libraries**: T023-T033 (mixed parallel/sequential, ~8 hours)
4. **MCP server**: T034-T038 (sequential, ~4 hours)
5. **Integration**: T039-T042 (sequential, ~3 hours)
6. **Polish**: T043-T050 (parallel, ~4 hours)

**Total estimated**: ~24 hours of implementation time

## Notes

- Each Rust library has its own Cargo.toml and can be developed independently
- Python components use async/await throughout for performance
- All database operations use prepared statements
- Contract tests use real PostgreSQL via testcontainers
- Performance benchmarks must meet constitution requirements (p95 <600ms)
- Citation format must include file:line for every result

## Validation Checklist

*GATE: All must pass before implementation*

- [x] All 11 MCP tools have contract tests (T012-T019)
- [x] All 10 entities have corresponding tables (T007-T009)
- [x] All tests come before implementation (Phase 3.3 before 3.4)
- [x] Parallel tasks operate on different files
- [x] Each task specifies exact file paths
- [x] No parallel tasks modify the same file
- [x] TDD cycle enforced (tests must fail first)

---

*Generated from implementation plan v1.0.0 - Ready for execution*
