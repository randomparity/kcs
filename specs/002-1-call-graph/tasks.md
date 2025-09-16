# Tasks: Call Graph Extraction

**Input**: Design documents from `/home/dave/src/kcs/specs/002-1-call-graph/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓)

## Execution Flow (main)

```text
1. Load plan.md from feature directory ✓
   → Tech stack: Rust 1.75, Python 3.11, tree-sitter, tokio, serde
   → Structure: Single project (Rust workspace + Python package)
2. Load optional design documents ✓:
   → data-model.md: CallEdge, CallType entities → model tasks
   → contracts/: parser-api.json, mcp-endpoints.json → contract test tasks
   → research.md: Tree-sitter AST traversal decisions → setup tasks
3. Generate tasks by category ✓:
   → Setup: Rust struct definitions, test fixtures
   → Tests: contract tests, integration tests (5 scenarios)
   → Core: AST traversal, call extraction, database integration
   → Integration: MCP endpoint updates, database persistence
   → Polish: performance tests, benchmarks
4. Apply task rules ✓:
   → Different files = marked [P] for parallel
   → Same file modifications = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially T001-T022
6. Generate dependency graph ✓
7. Create parallel execution examples ✓
8. Validate task completeness ✓
   → All contracts have tests ✓
   → All entities have models ✓
   → All endpoints implemented ✓
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup

- [x] T001 [P] Create CallEdge and CallType structs in src/rust/kcs-parser/src/types.rs
- [x] T002 [P] Extend ParseResult struct to include call_edges field in src/rust/kcs-parser/src/lib.rs
- [x] T003 [P] Create test fixtures directory and sample C files in tests/fixtures/call_graph/

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

### CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation

- [x] T004 [P] Contract test for parse_file with call graph extraction in tests/contract/test_parser_api.rs
- [x] T005 [P] Contract test for who_calls MCP endpoint in tests/contract/test_mcp_who_calls.py
- [x] T006 [P] Contract test for list_dependencies MCP endpoint in tests/contract/test_mcp_dependencies.py
- [x] T007 [P] Contract test for entrypoint_flow MCP endpoint in tests/contract/test_mcp_entrypoint.py
- [x] T008 [P] Contract test for impact_of MCP endpoint in tests/contract/test_mcp_impact.py
- [x] T009 [P] Integration test for basic call extraction in tests/integration/test_simple_calls.rs
- [x] T010 [P] Integration test for function pointer calls in tests/integration/test_function_pointers.rs
- [x] T011 [P] Integration test for MCP endpoint integration in tests/integration/test_mcp_integration.py
- [x] T012 [P] Performance test for large file parsing in tests/performance/test_call_graph_perf.rs

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [x] T013 Add tree-sitter call detection in src/rust/kcs-parser/src/call_extractor.rs
- [x] T014 Implement AST traversal for call_expression nodes in src/rust/kcs-parser/src/call_extractor.rs
- [x] T015 Integrate call extraction with main parsing loop in src/rust/kcs-parser/src/lib.rs
- [x] T016 [P] Add CLI flag --include-calls to parser binary in src/rust/kcs-parser/src/main.rs
- [ ] T017 [P] Update JSON output format to include call_edges in src/rust/kcs-parser/src/output.rs

## Phase 3.4: Integration

- [ ] T018 Update database schema to populate call_edges table in src/python/kcs_mcp/database.py
- [ ] T019 Modify who_calls MCP endpoint to use call graph data in src/python/kcs_mcp/endpoints/who_calls.py
- [ ] T020 Modify list_dependencies endpoint to use call graph data in src/python/kcs_mcp/endpoints/dependencies.py
- [ ] T021 Update entrypoint_flow and impact_of endpoints in src/python/kcs_mcp/endpoints/

## Phase 3.5: Polish

- [ ] T022 [P] Add performance benchmarks for call graph extraction in benches/call_graph_bench.rs

## Dependencies

- Setup (T001-T003) before tests (T004-T012)
- Tests (T004-T012) before implementation (T013-T017)
- Core implementation (T013-T017) before integration (T018-T021)
- T013-T014 must complete before T015 (same file)
- T018 blocks T019-T021 (database dependency)
- Everything before polish (T022)

## Parallel Example

```text
# Phase 3.1 - Setup tasks can run together:
Task: "Create CallEdge and CallType structs in src/rust/kcs-parser/src/types.rs"
Task: "Extend ParseResult struct in src/rust/kcs-parser/src/lib.rs"
Task: "Create test fixtures in tests/fixtures/call_graph/"

# Phase 3.2 - Contract tests can run together:
Task: "Contract test for parse_file in tests/contract/test_parser_api.rs"
Task: "Contract test for who_calls in tests/contract/test_mcp_who_calls.py"
Task: "Contract test for list_dependencies in tests/contract/test_mcp_dependencies.py"
Task: "Contract test for entrypoint_flow in tests/contract/test_mcp_entrypoint.py"
Task: "Contract test for impact_of in tests/contract/test_mcp_impact.py"

# Integration tests can run in parallel:
Task: "Integration test for basic calls in tests/integration/test_simple_calls.rs"
Task: "Integration test for function pointers in tests/integration/test_function_pointers.rs"
Task: "Integration test for MCP endpoints in tests/integration/test_mcp_integration.py"
```

## Notes

- [P] tasks = different files, no dependencies
- Verify tests fail before implementing (constitutional requirement)
- Commit after each task with proper commit messages
- Address lint/format/typecheck issues via pre-commit hooks
- Use existing project patterns (tree-sitter, tokio, pytest)

## Task Generation Rules Applied

1. **From Contracts**:
   - parser-api.json → T004 contract test [P]
   - mcp-endpoints.json → T005-T008 contract tests [P]
   - Each endpoint → T019-T021 implementation tasks

2. **From Data Model**:
   - CallEdge entity → T001 model creation task [P]
   - CallType enum → T001 model creation task [P]
   - ParseResult extension → T002 [P]

3. **From Quickstart Scenarios**:
   - 5 test scenarios → T009-T012 integration tests [P]
   - Performance validation → T022 benchmark [P]

4. **From Research Decisions**:
   - Tree-sitter approach → T013-T014 AST traversal
   - Parallel processing → T015 integration
   - CLI interface → T016-T017 output tasks [P]

## Validation Checklist

### GATE: Checked before execution

- [x] All contracts have corresponding tests (T004-T008)
- [x] All entities have model tasks (T001-T002)
- [x] All tests come before implementation (T004-T012 before T013-T017)
- [x] Parallel tasks truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Constitutional TDD order enforced (RED-GREEN-Refactor)

## Ready for Execution

Tasks are ready for execution following TDD principles. Each task includes:

- Exact file paths for implementation
- Clear dependencies and ordering
- Parallel execution opportunities marked with [P]
- Constitutional compliance (tests before implementation)
- Integration with existing KCS architecture
