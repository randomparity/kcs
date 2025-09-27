# Tasks: Verify Document System

**Input**: Design documents from `/specs/009-verify-document-system/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/vectorstore-api.yaml, quickstart.md

## Execution Flow (main)

```
1. Load plan.md from feature directory
   → Extract: Python 3.11+, asyncpg, pgvector, pydantic
   → Structure: Single project for documentation/verification
2. Load design documents:
   → data-model.md: 4 entities (indexed_content, vector_embedding, search_query, search_result)
   → contracts/vectorstore-api.yaml: 9 API endpoints
   → research.md: Schema discrepancies to document
   → quickstart.md: Verification script with 6 acceptance criteria
3. Generate tasks by category:
   → Verification: Run acceptance criteria tests
   → Documentation: Generate OpenAPI docs, ERD diagrams
   → Validation: Test multiple chunks, verify dimensions
   → Resolution: Document discrepancies
4. Apply task rules:
   → Documentation tasks can run in parallel [P]
   → Verification must complete before documentation
   → Tests validate implementation correctness
5. Number tasks sequentially (T001, T002...)
6. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Implementation**: `src/python/semantic_search/database/vector_store.py`
- **Schema**: `src/sql/migrations/014_semantic_search_core.sql`
- **Documentation**: `docs/vectorstore/`
- **Tests**: `tests/verification/`

## Phase 3.1: Setup & Prerequisites

- [X] T001 Create verification test directory structure at tests/verification/
- [X] T002 Create documentation output directory at docs/vectorstore/
- [X] T003 [P] Install Python dependencies: asyncpg, pgvector, pydantic, pyyaml
- [X] T004 [P] Verify PostgreSQL connection and pgvector extension

## Phase 3.2: Verification Tests (MUST COMPLETE FIRST)

**CRITICAL: Run these to establish baseline before documentation**

- [X] T005 Run verification script from quickstart.md in tests/verification/test_verify_foundation.py
- [X] T006 Verify all 9 VectorStore API methods exist with correct signatures
- [X] T007 Verify database schema matches production migration (014_semantic_search_core.sql)
- [X] T008 Verify vector dimensions are 384 (not 768) in actual database
- [X] T009 Test multiple chunks per file functionality with actual inserts
- [X] T010 Validate all unique constraints and indexes exist in database
- [X] T010a Verify documentation against actual VectorStore implementation per FR-007

## Phase 3.3: Documentation Generation

- [X] T011 [P] Generate OpenAPI documentation from contracts/vectorstore-api.yaml to docs/vectorstore/api.html
- [X] T012 [P] Create ERD diagram from data-model.md to docs/vectorstore/schema.svg
- [X] T013 [P] Generate method signatures documentation in docs/vectorstore/methods.md
- [X] T014 [P] Create database column reference in docs/vectorstore/columns.md
- [X] T015 Document all discrepancies (API and schema) between intended design and actual implementation in docs/vectorstore/discrepancies.md

## Phase 3.4: Validation & Mapping

- [X] T016 Map Python DBIndexedContent fields to database columns in docs/vectorstore/field-mapping.md
- [X] T017 Map Python DBVectorEmbedding fields to database columns in docs/vectorstore/embedding-mapping.md
- [X] T018 Validate OpenAPI spec against actual VectorStore implementation
- [X] T019 Cross-reference constraints between code validation and database constraints
- [X] T020 Test similarity_search with 384-dimensional vectors

## Phase 3.5: Integration Documentation

- [X] T021 [P] Document VectorStore initialization and connection setup in docs/vectorstore/setup.md
- [X] T022 [P] Create usage examples for each API method in docs/vectorstore/examples.md
- [X] T023 [P] Document error handling patterns in docs/vectorstore/errors.md
- [X] T024 Generate comprehensive API reference combining all documentation

## Phase 3.6: Polish & Finalization

- [X] T025 [P] Create migration guide for schema discrepancies in docs/vectorstore/migration.md
- [X] T026 [P] Add performance notes for IVFFlat vs HNSW indexes in docs/vectorstore/performance.md
- [X] T027 Validate all documentation links and cross-references
- [X] T028 Run final verification script to ensure nothing broke
- [X] T029 Create summary report of verification results in docs/vectorstore/summary.md

## Dependencies

- Verification (T005-T010) must complete before documentation (T011-T015)
- T005 blocks all other verification tasks
- T016-T017 depend on T013-T014
- T018-T020 depend on T011
- T024 depends on T011-T023
- T028 must be last

## Parallel Execution Examples

### Verification Setup (T003-T004)

```
Task: "Install Python dependencies: asyncpg, pgvector, pydantic, pyyaml"
Task: "Verify PostgreSQL connection and pgvector extension"
```

### Documentation Generation (T011-T015)

```
Task: "Generate OpenAPI documentation from contracts/vectorstore-api.yaml"
Task: "Create ERD diagram from data-model.md"
Task: "Generate method signatures documentation"
Task: "Create database column reference"
```

### Usage Documentation (T021-T023)

```
Task: "Document VectorStore initialization and connection setup"
Task: "Create usage examples for each API method"
Task: "Document error handling patterns"
```

### Final Documentation (T025-T026)

```
Task: "Create migration guide for schema discrepancies"
Task: "Add performance notes for IVFFlat vs HNSW indexes"
```

## Notes

- This is a verification/documentation feature, not implementation
- Focus on documenting actual state, not intended state
- All discrepancies must be clearly documented
- Tests verify the system works as currently implemented
- Documentation should be immediately usable by developers

## Validation Checklist

*GATE: All must pass before marking feature complete*

- [x] All 9 VectorStore methods documented
- [x] All 4 database tables documented
- [x] OpenAPI spec covers all endpoints
- [x] ERD diagram shows all relationships
- [x] Multiple chunks per file verified working
- [x] 384-dimensional vectors confirmed
- [x] All discrepancies documented
- [x] Verification script runs successfully
