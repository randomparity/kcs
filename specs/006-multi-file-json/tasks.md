# Tasks: Multi-File JSON Output Strategy

**Input**: Design documents from `/specs/006-multi-file-json/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```text
1. Load plan.md from feature directory
   → Loaded: Rust + Python + SQL tech stack
   → Structure: Single project extending KCS
2. Load optional design documents:
   → data-model.md: 4 entities (ChunkManifest, ChunkMetadata, ChunkFile, ProcessingStatus)
   → contracts/: chunk-api.yaml with 4 endpoints
   → research.md: Technical decisions on chunking strategy
3. Generate tasks by category:
   → Setup: Database migration, test fixtures
   → Tests: Contract tests for 4 endpoints, integration tests
   → Core: Rust serializer, Python loader, database models
   → Integration: Shell script updates, chunk processor
   → Polish: Performance tests, documentation
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests? ✓
   → All entities have models? ✓
   → All endpoints implemented? ✓
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths shown below for KCS multi-language structure

## Phase 3.1: Setup

- [x] T001 Create database migration for chunk tracking tables in src/sql/migrations/012_chunk_tracking.sql
- [x] T002 [P] Create test fixture directory structure at tests/fixtures/chunks/
- [x] T003 [P] Generate sample chunk files for testing in tests/fixtures/chunks/sample_*.json

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

### Critical Requirement

These tests MUST be written and MUST FAIL before ANY implementation

### Contract Tests

- [x] T004 [P] Contract test GET /mcp/chunks/manifest in tests/contract/test_chunk_manifest.py
- [x] T005 [P] Contract test GET /mcp/chunks/{id}/status in tests/contract/test_chunk_status.py
- [x] T006 [P] Contract test POST /mcp/chunks/{id}/process in tests/contract/test_chunk_process.py
- [x] T007 [P] Contract test POST /mcp/chunks/process/batch in tests/contract/test_chunk_batch.py

### Integration Tests

- [x] T008 [P] Integration test full 60-chunk indexing scenario in tests/integration/test_full_chunking.py
- [x] T009 [P] Integration test resume from chunk 15 failure in tests/integration/test_resume_processing.py
- [x] T010 [P] Integration test incremental subsystem update in tests/integration/test_incremental_chunks.py
- [x] T011 [P] Integration test chunk boundary validation in tests/integration/test_chunk_boundaries.py

### Rust Tests

- [x] T012 [P] Unit test chunk writer with size limits in src/rust/kcs-serializer/src/chunk_writer_test.rs
- [x] T013 [P] Unit test manifest generation in src/rust/kcs-serializer/src/manifest_test.rs
- [x] T014 [P] Unit test SHA256 checksum calculation in src/rust/kcs-serializer/src/checksum_test.rs

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Database Layer

- [x] T015 Apply database migration 012_chunk_tracking.sql using psql
- [x] T016 [P] Create Pydantic models for chunks in src/python/kcs_mcp/models/chunk_models.py
- [x] T017 [P] Database queries for chunk operations in src/python/kcs_mcp/database/chunk_queries.py

### Rust Implementation

- [x] T018 [P] Implement ChunkWriter in src/rust/kcs-serializer/src/chunk_writer.rs
- [x] T019 [P] Implement ManifestBuilder in src/rust/kcs-serializer/src/manifest.rs
- [x] T020 [P] Add checksum module in src/rust/kcs-serializer/src/checksum.rs
- [x] T021 Update kcs-parser with --chunk-size flag in src/rust/kcs-parser/src/cli.rs
- [x] T022 Add chunk coordination to kcs-python-bridge in src/rust/kcs-python-bridge/src/lib.rs

### Python Implementation

- [x] T023 [P] Create chunk_loader module in src/python/kcs_mcp/chunk_loader.py
- [x] T024 [P] Implement chunk processing with resume in src/python/kcs_mcp/chunk_processor.py
- [x] T025 Add chunk endpoints to FastAPI in src/python/kcs_mcp/tools.py
- [ ] T026 [P] Add chunk status tracking in src/python/kcs_mcp/chunk_tracker.py

### Shell Scripts

- [ ] T027 Update index_kernel.sh with chunking flags in tools/index_kernel.sh
- [ ] T028 [P] Create process_chunks.py script in tools/process_chunks.py

## Phase 3.4: Integration

- [ ] T029 Connect chunk_loader to database connection pool
- [ ] T030 Add structured logging for chunk progress
- [ ] T031 Implement parallel chunk processing with semaphore
- [ ] T032 Add transaction boundaries per chunk
- [ ] T033 Integrate checksum verification before processing

## Phase 3.5: Polish

### Performance Tests

- [ ] T034 [P] Performance test 50MB chunk generation in tests/performance/test_chunk_generation.py
- [ ] T035 [P] Performance test parallel chunk loading in tests/performance/test_parallel_loading.py
- [ ] T036 [P] Memory usage test per chunk in tests/performance/test_memory_usage.py

### Documentation

- [ ] T037 [P] Update tools/README.md with chunking examples
- [ ] T038 [P] Add troubleshooting guide for chunk failures in docs/troubleshooting.md
- [ ] T039 [P] Update CLAUDE.md with chunk workflow (already done in plan phase)

### Validation

- [ ] T040 Run quickstart.md end-to-end test scenario
- [ ] T041 Verify all contract tests pass with implementation
- [ ] T042 Check performance meets targets (<30 min full index)

## Dependencies

- Database migration (T015) blocks model tasks (T016-T017)
- Tests (T004-T014) must fail before implementation (T018-T028)
- Rust chunk writer (T018) blocks Python loader (T023)
- Manifest builder (T019) blocks process_chunks.py (T028)
- Core implementation (T018-T028) before integration (T029-T033)
- Everything before polish (T034-T042)

## Parallel Execution Examples

### Launch all contract tests together (T004-T007)

```bash
Task agent="general-purpose" "Write contract test for GET /mcp/chunks/manifest endpoint in tests/contract/test_chunk_manifest.py using OpenAPI schema from contracts/chunk-api.yaml"
Task agent="general-purpose" "Write contract test for GET /mcp/chunks/{id}/status endpoint in tests/contract/test_chunk_status.py using OpenAPI schema"
Task agent="general-purpose" "Write contract test for POST /mcp/chunks/{id}/process endpoint in tests/contract/test_chunk_process.py using OpenAPI schema"
Task agent="general-purpose" "Write contract test for POST /mcp/chunks/process/batch endpoint in tests/contract/test_chunk_batch.py using OpenAPI schema"
```

### Launch Rust implementations in parallel (T018-T020)

```bash
Task agent="general-purpose" "Implement ChunkWriter with 50MB size limit and streaming JSON in src/rust/kcs-serializer/src/chunk_writer.rs"
Task agent="general-purpose" "Implement ManifestBuilder to generate chunk manifest with metadata in src/rust/kcs-serializer/src/manifest.rs"
Task agent="general-purpose" "Add SHA256 checksum calculation module in src/rust/kcs-serializer/src/checksum.rs"
```

### Launch Python modules in parallel (T023-T024, T026)

```bash
Task agent="general-purpose" "Create chunk_loader module with async chunk reading in src/python/kcs_mcp/chunk_loader.py"
Task agent="general-purpose" "Implement chunk processor with resume capability in src/python/kcs_mcp/chunk_processor.py"
Task agent="general-purpose" "Add chunk status tracking with database persistence in src/python/kcs_mcp/chunk_tracker.py"
```

## Notes

- [P] tasks = different files, no dependencies
- Verify tests fail before implementing (RED phase of TDD)
- Commit after each task with descriptive message
- Run `make check` before moving to next task
- Avoid modifying same file in parallel tasks

## Task Generation Rules

### Applied during main() execution

1. **From Contracts**:
   - chunk-api.yaml → 4 contract test tasks [P]
   - 4 endpoints → 4 implementation tasks in tools.py

2. **From Data Model**:
   - ChunkManifest → manifest.rs implementation
   - ChunkMetadata → chunk_writer.rs implementation
   - ProcessingStatus → chunk_tracker.py implementation
   - Database schema → migration task

3. **From User Stories**:
   - 60-chunk scenario → full indexing integration test
   - Resume capability → failure recovery test
   - Incremental updates → subsystem update test

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist

### GATE: Checked by main() before returning

- [x] All contracts have corresponding tests (T004-T007)
- [x] All entities have model tasks (T016, T018-T020)
- [x] All tests come before implementation (T004-T014 before T015-T028)
- [x] Parallel tasks truly independent (verified file paths)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task

---

### Ready for execution - 42 tasks total
