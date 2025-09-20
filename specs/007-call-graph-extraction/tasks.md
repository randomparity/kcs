# Tasks: Call Graph Extraction Specification

**Input**: Design documents from `/specs/007-call-graph-extraction/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```text
1. Load plan.md from feature directory
   → Extracted: Rust 1.75+, Tree-sitter, petgraph, PostgreSQL, Python MCP
   → Structure: Single project, existing KCS codebase extension
2. Load optional design documents:
   → data-model.md: 6 entities (CallEdge, Function, CallSite, MacroCall, FunctionPointer, CallPath)
   → contracts/: call-graph-api.yaml with 4 MCP endpoints
   → research.md: Tree-sitter query patterns, performance optimization decisions
3. Generate tasks by category:
   → Setup: Rust dependencies, database schema, Tree-sitter queries
   → Tests: 4 contract tests, 4 integration tests based on quickstart scenarios
   → Core: 6 entity models, AST extraction logic, MCP endpoint implementations
   → Integration: Database operations, graph algorithms, MCP protocol integration
   → Polish: Performance optimization, unit tests, documentation
4. Apply task rules:
   → Different files/modules = mark [P] for parallel execution
   → Same file/module = sequential to avoid conflicts
   → Tests before implementation (TDD compliance)
5. Number tasks sequentially (T001-T040)
6. Generate dependency graph showing Tree-sitter → extraction → endpoints flow
7. Create parallel execution examples for independent tasks
8. Validate task completeness:
   → All 4 contracts have tests ✓
   → All 6 entities have models ✓
   → All endpoints implemented ✓
9. Return: SUCCESS (40 tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/rust/kcs-*/src/`, `tests/` at repository root
- Paths assume existing KCS codebase structure from plan.md

## Phase 3.1: Setup

- [x] T001 Add Tree-sitter call extraction dependencies to src/rust/kcs-parser/Cargo.toml
- [x] T002 Create database migration 013_call_graph_tables.sql with call_edges,
  function_pointers, macro_calls tables
- [x] T003 [P] Create Tree-sitter query file src/rust/kcs-parser/queries/call_patterns.scm
  for C call detection
- [x] T004 [P] Configure Rust clippy and formatting for call graph modules

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

### CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation

### Contract Tests (API Endpoints)

- [x] T005 [P] Contract test POST /mcp/tools/extract_call_graph in tests/contract/test_call_graph_extraction.py
- [x] T006 [P] Contract test POST /mcp/tools/get_call_relationships in tests/contract/test_call_relationships.py
- [x] T007 [P] Contract test POST /mcp/tools/trace_call_path in tests/contract/test_call_path_tracing.py
- [x] T008 [P] Contract test POST /mcp/tools/analyze_function_pointers in tests/contract/test_function_pointers.py

### Integration Tests (Quickstart Scenarios)

- [x] T009 [P] Integration test direct function call extraction in tests/integration/test_direct_calls.py
- [x] T010 [P] Integration test function relationship queries in tests/integration/test_relationship_queries.py
- [x] T011 [P] Integration test call path tracing in tests/integration/test_path_tracing.py
- [x] T012 [P] Integration test function pointer analysis in tests/integration/test_pointer_analysis.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models and Database Schema

- [x] T013 [P] CallEdge struct in src/rust/kcs-graph/src/call_edge.rs
- [x] T014 [P] CallSite struct in src/rust/kcs-graph/src/call_site.rs
- [x] T015 [P] MacroCall struct in src/rust/kcs-graph/src/macro_call.rs
- [x] T016 [P] FunctionPointer struct in src/rust/kcs-graph/src/function_pointer.rs
- [x] T017 [P] CallPath struct in src/rust/kcs-graph/src/call_path.rs
- [x] T018 [P] CallType and ConfidenceLevel enums in src/rust/kcs-graph/src/types.rs

### Tree-sitter AST Extraction

- [x] T019 Direct function call detection in src/rust/kcs-parser/src/call_extraction/direct_calls.rs
- [x] T020 Function pointer call detection in src/rust/kcs-parser/src/call_extraction/pointer_calls.rs
- [x] T021 Macro call expansion in src/rust/kcs-parser/src/call_extraction/macro_calls.rs
- [x] T022 Callback pattern recognition in src/rust/kcs-parser/src/call_extraction/callbacks.rs
- [x] T023 Conditional compilation handling in src/rust/kcs-parser/src/call_extraction/conditional.rs

### Core Call Graph Engine

- [x] T024 CallExtractor main engine in src/rust/kcs-parser/src/call_extraction/mod.rs
- [x] T025 AST traversal coordinator in src/rust/kcs-parser/src/ast_traversal.rs
- [x] T026 Call classification logic in src/rust/kcs-parser/src/call_classifier.rs
- [x] T027 Performance optimization with rayon parallelization in src/rust/kcs-parser/src/parallel_processing.rs

## Phase 3.4: Integration

### Database Operations

- [x] T028 [P] Database insertion functions for call_edges table in src/python/kcs_mcp/database/call_graph.py
- [x] T029 [P] Query functions for call relationships in src/python/kcs_mcp/database/queries.py
- [x] T030 Graph traversal algorithms integration with petgraph in src/rust/kcs-graph/src/traversal.rs

### MCP Protocol Endpoints

- [x] T031 Extract call graph MCP tool in src/python/kcs_mcp/tools/extract_call_graph.py
- [ ] T032 Get call relationships MCP tool in src/python/kcs_mcp/tools/get_call_relationships.py
- [ ] T033 Trace call path MCP tool in src/python/kcs_mcp/tools/trace_call_path.py
- [ ] T034 Analyze function pointers MCP tool in src/python/kcs_mcp/tools/analyze_function_pointers.py

### System Integration

- [ ] T035 Python-Rust bridge integration in src/python/kcs_mcp/rust_bridge.py
- [ ] T036 Error handling and logging across call graph pipeline
- [ ] T037 Configuration management for call extraction settings

## Phase 3.5: Polish

- [ ] T038 [P] Unit tests for Tree-sitter queries in tests/unit/test_call_patterns.py
- [ ] T039 [P] Performance benchmarks for large kernel extraction in tests/performance/test_call_graph_performance.py
- [ ] T040 [P] Update CLAUDE.md with call graph extraction capabilities

## Dependencies

### Critical Path

- T002 (DB schema) → T013-T018 (models) → T028-T029 (DB operations)
- T003 (Tree-sitter queries) → T019-T023 (AST extraction) → T024-T027 (engine)
- T024 (engine) → T031-T034 (MCP endpoints)
- T005-T012 (tests) before T013-T037 (implementation)

### Parallel Groups

- Contract tests (T005-T008) can run together
- Integration tests (T009-T012) can run together
- Data models (T013-T018) can run together
- AST extraction modules (T019-T023) are sequential (shared Tree-sitter state)
- MCP tools (T031-T034) are sequential (shared Python modules)

## Parallel Example

```bash
# Launch contract tests together:
Task: "Contract test POST /mcp/tools/extract_call_graph in tests/contract/test_call_graph_extraction.py"
Task: "Contract test POST /mcp/tools/get_call_relationships in tests/contract/test_call_relationships.py"
Task: "Contract test POST /mcp/tools/trace_call_path in tests/contract/test_call_path_tracing.py"
Task: "Contract test POST /mcp/tools/analyze_function_pointers in tests/contract/test_function_pointers.py"

# Launch data models together:
Task: "CallEdge struct in src/rust/kcs-graph/src/call_edge.rs"
Task: "CallSite struct in src/rust/kcs-graph/src/call_site.rs"
Task: "MacroCall struct in src/rust/kcs-graph/src/macro_call.rs"
Task: "FunctionPointer struct in src/rust/kcs-graph/src/function_pointer.rs"
```

## Notes

- [P] tasks = different files/modules, no dependencies
- Verify tests fail before implementing (TDD compliance)
- Commit after each task completion
- Avoid: vague tasks, same file conflicts
- Tree-sitter query tasks (T019-T023) must be sequential due to shared parser state
- MCP endpoint tasks (T031-T034) must be sequential due to shared Python module structure

## Task Generation Rules

### Applied during main() execution

1. **From Contracts**:
   - call-graph-api.yaml → 4 contract test tasks [P] (T005-T008)
   - 4 endpoints → 4 implementation tasks (T031-T034)

2. **From Data Model**:
   - 6 entities → 6 model creation tasks [P] (T013-T018)
   - Database schema → migration task (T002)

3. **From Quickstart Scenarios**:
   - 4 scenarios → 4 integration tests [P] (T009-T012)
   - Performance requirements → benchmark tasks (T039)

4. **From Research Decisions**:
   - Tree-sitter queries → pattern detection tasks (T003, T019-T023)
   - Parallel processing → optimization tasks (T027)

## Validation Checklist

### GATE: Checked by main() before returning

- [x] All 4 contracts have corresponding tests (T005-T008)
- [x] All 6 entities have model tasks (T013-T018)
- [x] All tests come before implementation (T005-T012 before T013-T037)
- [x] Parallel tasks truly independent (different files/modules)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Critical dependencies identified (Tree-sitter → extraction → endpoints)
- [x] TDD compliance enforced (tests MUST fail before implementation)

## Performance Targets

- T002-T040 completion: 2-3 weeks
- T019-T027 (core extraction): 1 week
- T031-T034 (MCP integration): 3-4 days
- T038-T040 (polish): 2-3 days

### All 40 tasks ready for execution - system prepared for comprehensive call graph extraction capability
