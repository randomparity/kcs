# Tasks: Enhanced Kernel Entry Point and Symbol Detection

**Input**: Design documents from `/specs/004-current-code-has/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```text
1. Load plan.md from feature directory
   → Extracted: Rust 1.75, Python 3.11, existing KCS project
   → Structure: src/rust/, src/python/, tests/
2. Load optional design documents:
   → data-model.md: EntryPoint, Symbol, KernelPattern entities
   → contracts/: 3 endpoints → 3 contract test tasks
   → research.md: Partial implementations to enhance
3. Generate tasks by category:
   → Setup: DB migration for metadata columns
   → Tests: 3 contract tests, 5 integration tests
   → Core: Pattern detection, entry points, Clang bridge
   → Integration: Python bindings, MCP updates
   → Polish: Performance validation
4. Apply task rules:
   → Contract tests marked [P] (different files)
   → Integration tests marked [P] (independent)
   → Rust implementations sequential (same crate)
5. Number tasks sequentially (T001-T026)
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
- Rust code: `src/rust/kcs-*/src/`
- Python code: `src/python/kcs_mcp/`
- Tests: `tests/contract/`, `tests/integration/`, `tests/performance/`

## Phase 3.1: Setup

- [x] T001 Create database migration for metadata columns in `src/sql/migrations/005_metadata_columns.sql`
  - Add metadata JSONB to entry_point and symbol tables
  - Create kernel_pattern table
  - Add performance indexes

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

### Critical TDD Requirement

These tests MUST be written and MUST FAIL before ANY implementation

### Contract Tests

- [x] T002 [P] Contract test for extract_entry_points endpoint in `tests/contract/test_extract_entry_points.py`
  - Validate request/response schemas match `contracts/extract_entry_points.yaml`
  - Test with kernel fixture data
  - Verify new entry types returned (procfs, debugfs, netlink, etc.)
  - MUST FAIL initially (new types not implemented)

- [x] T003 [P] Contract test for detect_patterns endpoint in `tests/contract/test_detect_patterns.py`
  - Validate request/response schemas match `contracts/detect_patterns.yaml`
  - Test pattern detection on sample files
  - Verify EXPORT_SYMBOL, module_param detection
  - MUST FAIL initially (patterns not implemented)

- [x] T004 [P] Contract test for enhance_symbols endpoint in `tests/contract/test_enhance_symbols.py`
  - Validate request/response schemas match `contracts/enhance_symbols.yaml`
  - Test symbol enhancement with and without Clang
  - Verify metadata enrichment
  - MUST FAIL initially (Clang not integrated)

### Integration Tests

- [x] T005 [P] Integration test for complete entry point extraction in `tests/integration/test_entry_point_extraction.py`
  - Test all entry point types on kernel fixtures
  - Verify counts match expected ranges
  - Check metadata population

- [x] T006 [P] Integration test for pattern detection accuracy in `tests/integration/test_pattern_detection.py`
  - Test EXPORT_SYMBOL variants detection
  - Test module_param detection with descriptions
  - Verify false positive rate <1%

- [x] T007 [P] Integration test for Clang symbol enhancement in `tests/integration/test_clang_enhancement.py`
  - Test with mock compile_commands.json
  - Verify type information extraction
  - Test graceful degradation without Clang

- [x] T008 [P] Integration test for ioctl command extraction in `tests/integration/test_ioctl_detection.py`
  - Test ioctl magic number detection
  - Verify ioctl handler identification
  - Check command metadata storage

- [x] T009 [P] Integration test for subsystem analysis in `tests/integration/test_subsystem_analysis.py`
  - Test complete ext4 subsystem as in quickstart
  - Verify entry points, exports, and params detected
  - Check performance <30s for subsystem

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Pattern Detection (Rust)

- [x] T010 Implement EXPORT_SYMBOL pattern detection in `src/rust/kcs-parser/src/kernel_patterns.rs`
  - Add regex patterns for EXPORT_SYMBOL/GPL/NS
  - Extract symbol name and export type
  - Handle preprocessor conditionals

- [x] T011 Implement module_param pattern detection in `src/rust/kcs-parser/src/kernel_patterns.rs`
  - Add patterns for module_param/module_param_array
  - Extract parameter name, type, description
  - Associate MODULE_PARM_DESC with params

- [ ] T012 Implement boot parameter patterns in `src/rust/kcs-parser/src/kernel_patterns.rs`
  - Add __setup, early_param, core_param patterns
  - Extract parameter names and handlers

### Entry Point Extraction (Rust)

- [ ] T013 Enhance ioctl detection in `src/rust/kcs-extractor/src/ioctls.rs`
  - Implement ioctl command extraction
  - Detect _IO/_IOR/_IOW/_IOWR macros
  - Extract magic numbers and command codes

- [ ] T014 Implement procfs entry point detection in `src/rust/kcs-extractor/src/entry_points.rs`
  - Add patterns for proc_create variants
  - Extract proc_ops structures
  - Identify show/write handlers

- [ ] T015 Implement debugfs entry point detection in `src/rust/kcs-extractor/src/entry_points.rs`
  - Add patterns for debugfs_create functions
  - Extract file operations
  - Identify debugfs attributes

- [ ] T016 Implement netlink handler detection in `src/rust/kcs-extractor/src/entry_points.rs`
  - Add patterns for netlink_kernel_create
  - Extract message handlers
  - Identify protocol families

- [ ] T017 Implement interrupt handler detection in `src/rust/kcs-extractor/src/entry_points.rs`
  - Add patterns for request_irq variants
  - Extract IRQ numbers and handlers
  - Identify interrupt types

### Clang Integration (Rust)

- [ ] T018 Initialize Clang index in `src/rust/kcs-parser/src/clang_bridge.rs`
  - Implement ClangBridge::new() with index creation
  - Parse compile_commands.json
  - Handle missing compilation database gracefully

- [ ] T019 Implement symbol type extraction in `src/rust/kcs-parser/src/clang_bridge.rs`
  - Extract function signatures with return types
  - Get parameter names and types
  - Extract attributes (__init,__exit, etc.)

- [ ] T020 Implement documentation extraction in `src/rust/kcs-parser/src/clang_bridge.rs`
  - Extract kernel-doc comments
  - Parse function descriptions
  - Associate docs with symbols

## Phase 3.4: Integration

### Python Bindings

- [ ] T021 Update Python bindings for pattern detection in `src/rust/kcs-python-bridge/src/lib.rs`
  - Export detect_patterns function to Python
  - Handle pattern type filtering
  - Return structured pattern data

- [ ] T022 Update Python bindings for enhanced extraction in `src/rust/kcs-python-bridge/src/lib.rs`
  - Export new entry point types
  - Include metadata in results
  - Support Clang enablement flag

### Database Integration

- [ ] T023 Update database insertion for metadata in `src/python/kcs_mcp/database.py`
  - Modify insert_entry_points() to handle metadata JSONB
  - Modify insert_symbols() to handle metadata JSONB
  - Add insert_kernel_patterns() method

### CLI Updates

- [ ] T024 Update extract CLI for new options in `tools/extract_entry_points_streaming.py`
  - Add --pattern-detection flag
  - Add --enable-clang flag
  - Add --entry-types filter
  - Stream new entry point types

## Phase 3.5: Polish

- [ ] T025 [P] Performance validation in `tests/performance/test_pattern_performance.py`
  - Benchmark pattern detection speed
  - Test with full kernel directory
  - Verify <100ms per file target
  - Test parallel processing scalability

- [ ] T026 Update documentation in `docs/features/kernel_patterns.md`
  - Document new entry point types
  - Explain pattern detection
  - Provide Clang setup guide
  - Include performance tips

## Dependencies

- T001 (migration) blocks all database operations
- T002-T009 (tests) must complete and FAIL before T010-T020
- T010-T012 (patterns) can proceed in parallel after tests
- T013-T017 (entry points) can proceed in parallel after tests
- T018-T020 (Clang) can proceed after tests
- T021-T022 (bindings) require T010-T020 complete
- T023 requires T001 complete
- T024 requires T021-T022 complete
- T025-T026 only after all implementation complete

## Parallel Execution Examples

### Launch all contract tests together

```bash
# Using Task agents or pytest
Task: "Contract test for extract_entry_points in tests/contract/test_extract_entry_points.py"
Task: "Contract test for detect_patterns in tests/contract/test_detect_patterns.py"
Task: "Contract test for enhance_symbols in tests/contract/test_enhance_symbols.py"
```

### Launch all integration tests together

```bash
# These test different aspects independently
Task: "Integration test for entry point extraction in tests/integration/test_entry_point_extraction.py"
Task: "Integration test for pattern detection in tests/integration/test_pattern_detection.py"
Task: "Integration test for Clang enhancement in tests/integration/test_clang_enhancement.py"
Task: "Integration test for ioctl detection in tests/integration/test_ioctl_detection.py"
Task: "Integration test for subsystem analysis in tests/integration/test_subsystem_analysis.py"
```

### Parallel Rust implementation (after tests)

```bash
# Different modules can be developed in parallel
Task: "Implement EXPORT_SYMBOL patterns in kernel_patterns.rs"
Task: "Enhance ioctl detection in ioctls.rs"
Task: "Initialize Clang index in clang_bridge.rs"
```

## Validation Checklist

- [x] All 3 contracts have test tasks (T002-T004)
- [x] All entities have implementation (metadata via T001, patterns via T010-T012)
- [x] All 3 endpoints have contract tests
- [x] Pattern detection covered (T010-T012)
- [x] Entry point types covered (T013-T017)
- [x] Clang integration included (T018-T020)
- [x] Performance validation included (T025)
- [x] TDD order enforced (tests before implementation)
- [x] Parallel markers on independent tasks
- [x] File paths specific and accurate

## Commit Strategy

Follow RED-GREEN-Refactor cycle:

1. **RED Phase**: Commit T001-T009 (migration + all tests failing)
   - Commit message: "test(patterns): add failing tests for kernel pattern detection"

2. **GREEN Phase**: Commit T010-T024 (make tests pass)
   - T010-T012: "feat(patterns): implement kernel pattern detection"
   - T013-T017: "feat(entry-points): add comprehensive entry point detection"
   - T018-T020: "feat(clang): implement symbol enhancement with Clang"
   - T021-T024: "feat(integration): update Python bindings and CLI"

3. **REFACTOR Phase**: Commit T025-T026
   - "perf(patterns): validate and optimize pattern detection performance"
   - "docs(patterns): document kernel pattern detection features"

## Notes

- Pattern detection should be conservative to avoid false positives
- Clang integration must gracefully degrade when unavailable
- All new entry point types need test fixtures
- Performance is critical - use parallelism where possible
- Metadata JSONB allows future extensibility without schema changes
