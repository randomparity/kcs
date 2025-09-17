# Tasks: Implement MCP Tools

**Input**: Design documents from `/specs/003-implement-mcp-tools/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```text
1. Load plan.md from feature directory
   → Extracted: Python 3.11, FastAPI, asyncpg, PostgreSQL
   → Structure: Existing KCS project (src/python/, tests/)
2. Load optional design documents:
   → data-model.md: Using existing DB entities (no new models)
   → contracts/: 4 endpoint contracts → 4 contract test tasks
   → research.md: Database traversal strategies identified
3. Generate tasks by category:
   → Setup: None needed (existing project)
   → Tests: 4 contract tests, 4 integration tests
   → Core: 2 DB enhancements, 4 endpoint implementations
   → Integration: None needed (existing infrastructure)
   → Polish: 1 performance validation
4. Apply task rules:
   → Contract tests marked [P] (different files)
   → Integration tests marked [P] (independent)
   → DB methods sequential (same file)
   → Endpoint implementations sequential (same file)
5. Number tasks sequentially (T001-T015)
6. Generate dependency graph (tests → implementation)
7. Create parallel execution examples
8. Validate task completeness: ✓
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- Using existing KCS project structure
- Python code: `src/python/kcs_mcp/`
- Tests: `tests/contract/`, `tests/integration/`, `tests/performance/`
- Database code: `src/python/kcs_mcp/database.py`
- Endpoints: `src/python/kcs_mcp/tools.py`

## Phase 3.1: Setup

No setup tasks needed - using existing project infrastructure.

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation.

### Contract Tests

- [x] T001 [P] Contract test for who_calls endpoint in `tests/contract/test_mcp_who_calls_implementation.py`
  - Validate request/response schemas match `contracts/who_calls.yaml`
  - Test with empty DB (should return empty list)
  - Test with populated DB (should return actual callers)
  - MUST FAIL initially (currently returns mock data)

- [x] T002 [P] Contract test for list_dependencies endpoint in `tests/contract/test_mcp_list_dependencies_implementation.py`
  - Validate request/response schemas match `contracts/list_dependencies.yaml`
  - Test with empty DB (should return empty list)
  - Test with populated DB (should return actual callees)
  - MUST FAIL initially (currently returns mock data)

- [x] T003 [P] Contract test for entrypoint_flow endpoint in `tests/contract/test_mcp_entrypoint_flow_implementation.py`
  - Validate request/response schemas match `contracts/entrypoint_flow.yaml`
  - Test syscall mapping (__NR_read → sys_read)
  - Test flow steps generation
  - MUST FAIL initially (currently returns mock data)

- [x] T004 [P] Contract test for impact_of endpoint in `tests/contract/test_mcp_impact_of_implementation.py`
  - Validate request/response schemas match `contracts/impact_of.yaml`
  - Test with symbols, files, and diff inputs
  - Verify all required fields in response
  - MUST FAIL initially (currently returns mock data)

### Integration Tests

- [x] T005 [P] Integration test for citation accuracy in `tests/integration/test_mcp_citations.py`
  - Verify all responses include valid Span objects
  - Check file paths exist in test data
  - Validate line numbers are positive
  - Test SHA values are present

- [x] T006 [P] Integration test for depth limiting in `tests/integration/test_mcp_depth_traversal.py`
  - Test who_calls with depth=1, 3, 5
  - Test list_dependencies with depth=1, 3, 5
  - Verify results expand with depth
  - Ensure depth=5 is enforced as maximum

- [x] T007 [P] Integration test for cycle detection in `tests/integration/test_mcp_cycle_handling.py`
  - Create test data with circular dependencies
  - Verify traversal doesn't infinite loop
  - Check visited tracking works correctly
  - Test both who_calls and list_dependencies

- [x] T008 [P] Integration test for empty results handling in `tests/integration/test_mcp_empty_results.py`
  - Test with non-existent symbols
  - Test with empty database
  - Verify graceful degradation (empty lists, not errors)
  - Test all four endpoints

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Database Enhancements

- [x] T009 Enhance find_callers() method in `src/python/kcs_mcp/database.py`
  - Add recursive CTE for depth traversal (currently depth ignored)
  - Add visited tracking to prevent cycles
  - Implement depth parameter properly
  - Keep LIMIT 100 for performance

- [x] T010 Enhance find_callees() method in `src/python/kcs_mcp/database.py`
  - Add recursive CTE for depth traversal (currently depth ignored)
  - Add visited tracking to prevent cycles
  - Implement depth parameter properly
  - Keep LIMIT 100 for performance

### Endpoint Implementations

- [ ] T011 Implement who_calls endpoint in `src/python/kcs_mcp/tools.py:who_calls()`
  - Remove mock data fallback (lines 273-294)
  - Use enhanced find_callers() with depth
  - Ensure all results have citations
  - Handle empty results gracefully

- [ ] T012 Implement list_dependencies endpoint in `src/python/kcs_mcp/tools.py:list_dependencies()`
  - Remove mock data fallback (lines 348-370)
  - Use enhanced find_callees() with depth
  - Ensure all results have citations
  - Handle empty results gracefully

- [ ] T013 Implement entrypoint_flow endpoint in `src/python/kcs_mcp/tools.py:entrypoint_flow()`
  - Extend entry_to_syscall mapping (add more syscalls)
  - Implement proper flow tracing with visited set
  - Remove mock data fallback (lines 464-482)
  - Add support for ioctl and file_ops entry points

- [ ] T014 Implement impact_of endpoint in `src/python/kcs_mcp/tools.py:impact_of()`
  - Enhance symbol extraction from diff
  - Implement bidirectional traversal (callers + callees)
  - Add subsystem detection from symbol patterns
  - Calculate risk based on blast radius size

## Phase 3.4: Integration

No integration tasks needed - existing infrastructure handles DB connections, logging, and middleware.

## Phase 3.5: Polish

- [ ] T015 Performance validation in `tests/performance/test_mcp_performance.py`
  - Measure query times for all endpoints
  - Test with production-scale data (if available)
  - Verify p95 < 600ms constitutional requirement
  - Test concurrent request handling

## Dependencies

- Contract tests (T001-T004) must complete and FAIL before implementation
- Integration tests (T005-T008) should be written alongside contract tests
- T009-T010 (DB enhancements) must complete before T011-T014 (endpoints)
- T011-T014 can proceed after database work
- T015 (performance) only after all implementation complete

## Parallel Execution Examples

### Launch all contract tests together

```bash
# Using Task agents or manual execution
Task: "Contract test for who_calls endpoint in tests/contract/test_mcp_who_calls_implementation.py"
Task: "Contract test for list_dependencies endpoint in tests/contract/test_mcp_list_dependencies_implementation.py"
Task: "Contract test for entrypoint_flow endpoint in tests/contract/test_mcp_entrypoint_flow_implementation.py"
Task: "Contract test for impact_of endpoint in tests/contract/test_mcp_impact_of_implementation.py"
```

### Launch all integration tests together

```bash
# These can run in parallel as they test different aspects
Task: "Integration test for citation accuracy in tests/integration/test_mcp_citations.py"
Task: "Integration test for depth limiting in tests/integration/test_mcp_depth_traversal.py"
Task: "Integration test for cycle detection in tests/integration/test_mcp_cycle_handling.py"
Task: "Integration test for empty results handling in tests/integration/test_mcp_empty_results.py"
```

## Validation Checklist

- [x] All 4 contracts have test tasks (T001-T004)
- [x] All entities use existing models (no new model tasks needed)
- [x] All 4 endpoints have implementation tasks (T011-T014)
- [x] Database enhancements included (T009-T010)
- [x] Performance validation included (T015)
- [x] TDD order enforced (tests before implementation)
- [x] Parallel markers on independent tasks
- [x] File paths specific and accurate

## Commit Strategy

Follow RED-GREEN-Refactor cycle strictly:

1. **RED Phase**: Commit T001-T008 (all tests failing)
   - Commit message: "test(mcp): add failing contract and integration tests for MCP tools"

2. **GREEN Phase**: Commit T009-T014 (make tests pass)
   - T009-T010: "feat(database): add depth traversal to find_callers and find_callees"
   - T011-T014: "feat(mcp): implement actual functionality for MCP tool endpoints"

3. **REFACTOR Phase**: Commit T015 and any cleanup
   - "perf(mcp): validate query performance meets constitutional requirements"

## Notes

- Mock data removal should be complete - no fallbacks remaining
- All database queries must use connection pooling (already configured)
- Citations are mandatory - every response needs file:line spans
- Performance is critical - enforce depth and result limits
- Configuration filtering is optional but should be tested
