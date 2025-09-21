# Tasks: Semantic Search Engine (kcs-search)

**Input**: Design documents from `/home/dave/src/kcs/specs/008-semantic-search-engine/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```text
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Python components**: `src/python/` following existing KCS structure
- **Tests**: `tests/` at repository root (integrating into existing KCS)
- Paths assume integration into existing KCS architecture

## Phase 3.1: Setup

- [x] T001 Create semantic search project structure in src/python/semantic_search/
- [x] T002 Initialize Python dependencies: sentence-transformers==2.7.0, pgvector==0.3.0, psycopg2-binary==2.9.9
- [x] T003 [P] Configure database schema with pgvector tables in src/python/semantic_search/schema.sql
- [x] T004 [P] Set up logging configuration following KCS patterns in src/python/semantic_search/logging_config.py

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

## Critical: TDD Requirements

These tests MUST be written and MUST FAIL before ANY implementation

### Contract Tests [P]

- [x] T005 [P] Contract test semantic_search MCP tool in tests/contract/test_semantic_search_tool.py
- [x] T006 [P] Contract test index_content MCP tool in tests/contract/test_index_content_tool.py
- [x] T007 [P] Contract test get_index_status MCP tool in tests/contract/test_index_status_tool.py

### Model Tests [P]

- [x] T008 [P] SearchQuery model test in tests/unit/test_search_query_model.py
- [x] T009 [P] VectorEmbedding model test in tests/unit/test_vector_embedding_model.py
- [x] T010 [P] SearchResult model test in tests/unit/test_search_result_model.py
- [x] T011 [P] IndexedContent model test in tests/unit/test_indexed_content_model.py

### Integration Tests [P]

- [x] T012 [P] Integration test: memory allocation search scenario in tests/integration/test_memory_allocation_search.py
- [x] T013 [P] Integration test: buffer overflow search scenario in tests/integration/test_buffer_overflow_search.py
- [x] T014 [P] Integration test: configuration filtering in tests/integration/test_config_filtering.py
- [x] T015 [P] Integration test: MCP endpoint error handling in tests/integration/test_mcp_error_handling.py

### Performance Tests [P]

- [x] T016 [P] Performance test: query response under 600ms in tests/performance/test_query_performance.py
- [x] T017 [P] Performance test: concurrent user handling in tests/performance/test_concurrent_users.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models [P]

- [x] T018 [P] SearchQuery model with validation in src/python/semantic_search/models/search_query.py
- [x] T019 [P] VectorEmbedding model with pgvector integration in src/python/semantic_search/models/vector_embedding.py
- [x] T020 [P] SearchResult model with ranking logic in src/python/semantic_search/models/search_result.py
- [x] T021 [P] IndexedContent model with status tracking in src/python/semantic_search/models/indexed_content.py

### Core Services [P]

- [x] T022 [P] EmbeddingService with BAAI/bge-small-en-v1.5 integration in src/python/semantic_search/services/embedding_service.py
- [x] T023 [P] QueryPreprocessor for text normalization in src/python/semantic_search/services/query_preprocessor.py
- [x] T024 [P] VectorSearchService with pgvector operations in src/python/semantic_search/services/vector_search_service.py
- [x] T025 [P] RankingService with hybrid BM25+semantic scoring in src/python/semantic_search/services/ranking_service.py

### Database Layer

- [x] T026 Database connection pool with PostgreSQL in src/python/semantic_search/database/connection.py
- [x] T027 Vector storage operations in src/python/semantic_search/database/vector_store.py
- [x] T028 Index management and optimization in src/python/semantic_search/database/index_manager.py

### CLI Interface [P]

- [x] T029 [P] CLI search command in src/python/semantic_search/cli/search_commands.py
- [x] T030 [P] CLI index command in src/python/semantic_search/cli/index_commands.py
- [x] T031 [P] CLI status command in src/python/semantic_search/cli/status_commands.py

## Phase 3.4: Integration

### MCP Endpoints

- [ ] T032 semantic_search MCP tool implementation in src/python/semantic_search/mcp/search_tool.py
- [ ] T033 index_content MCP tool implementation in src/python/semantic_search/mcp/index_tool.py
- [ ] T034 get_index_status MCP tool implementation in src/python/semantic_search/mcp/status_tool.py
- [ ] T035 MCP error handling and validation in src/python/semantic_search/mcp/error_handlers.py

### System Integration

- [ ] T036 Integrate with existing KCS logging infrastructure
- [ ] T037 Add semantic search to main KCS server startup
- [ ] T038 Database migration scripts for semantic search tables
- [ ] T039 Configuration management for embedding model paths

### Background Processing

- [ ] T040 Content indexing background worker in src/python/semantic_search/workers/indexing_worker.py
- [ ] T041 Incremental index updates based on file changes
- [ ] T042 Data retention policy enforcement per FR-012

## Phase 3.5: Polish

### Additional Tests [P]

- [ ] T043 [P] Unit tests for embedding service in tests/unit/test_embedding_service.py
- [ ] T044 [P] Unit tests for query preprocessing in tests/unit/test_query_preprocessor.py
- [ ] T045 [P] Unit tests for ranking algorithm in tests/unit/test_ranking_service.py
- [ ] T046 [P] Unit tests for vector operations in tests/unit/test_vector_operations.py

### Documentation and Optimization

- [ ] T047 [P] Create llms.txt documentation in docs/python/semantic_search/llms.txt
- [ ] T048 Performance optimization: query result caching
- [ ] T049 Performance optimization: vector index tuning
- [ ] T050 Execute quickstart validation scenarios from quickstart.md

### Final Validation

- [ ] T051 Run complete test suite and ensure all pass
- [ ] T052 Validate constitutional compliance (MCP-first, read-only, citations)
- [ ] T053 Performance benchmarking against requirements (p95 ≤ 600ms)

## Dependencies

### Phase Dependencies

- Setup (T001-T004) before all other phases
- Tests (T005-T017) before implementation (T018-T042)
- Models (T018-T021) before services (T022-T025)
- Services before integration (T032-T042)
- Core implementation before polish (T043-T053)

### Specific Dependencies

- T003 (schema) blocks T026 (database connection)
- T018-T021 (models) block T026-T028 (database operations)
- T022 (embedding service) blocks T032 (search tool)
- T024 (vector search) blocks T032 (search tool)
- T026 (database) blocks T038 (migrations)
- T032-T034 (MCP tools) block T037 (KCS integration)

## Parallel Execution Examples

### Contract Tests (can run simultaneously)

```
Task: "Contract test semantic_search MCP tool in tests/contract/test_semantic_search_tool.py"
Task: "Contract test index_content MCP tool in tests/contract/test_index_content_tool.py"
Task: "Contract test get_index_status MCP tool in tests/contract/test_index_status_tool.py"
```

### Model Tests (can run simultaneously)

```
Task: "SearchQuery model test in tests/unit/test_search_query_model.py"
Task: "VectorEmbedding model test in tests/unit/test_vector_embedding_model.py"
Task: "SearchResult model test in tests/unit/test_search_result_model.py"
Task: "IndexedContent model test in tests/unit/test_indexed_content_model.py"
```

### Core Services (can run simultaneously after models complete)

```
Task: "EmbeddingService with BAAI/bge-small-en-v1.5 integration in src/python/semantic_search/services/embedding_service.py"
Task: "QueryPreprocessor for text normalization in src/python/semantic_search/services/query_preprocessor.py"
Task: "VectorSearchService with pgvector operations in src/python/semantic_search/services/vector_search_service.py"
Task: "RankingService with hybrid BM25+semantic scoring in src/python/semantic_search/services/ranking_service.py"
```

## Notes

- [P] tasks = different files, no dependencies between them
- Verify tests fail before implementing (RED phase of TDD)
- Commit after each task completion
- Follow KCS constitutional requirements: MCP-first, read-only, citation-based
- CPU-only operation constraint for BAAI/bge-small-en-v1.5 model
- Performance target: p95 query response ≤ 600ms

## Task Generation Rules

### Application Rules

Applied during main() execution

1. **From Contracts**:
   - mcp-search-tools.json → 3 contract test tasks [P] (T005-T007)
   - Each MCP tool → implementation task (T032-T034)

2. **From Data Model**:
   - SearchQuery entity → model task [P] (T018)
   - VectorEmbedding entity → model task [P] (T019)
   - SearchResult entity → model task [P] (T020)
   - IndexedContent entity → model task [P] (T021)

3. **From User Stories** (quickstart.md):
   - Memory allocation search → integration test [P] (T012)
   - Buffer overflow search → integration test [P] (T013)
   - Configuration filtering → integration test [P] (T014)
   - Performance validation → performance tests [P] (T016-T017)

4. **From Technical Context** (plan.md):
   - BAAI/bge-small-en-v1.5 → embedding service (T022)
   - pgvector → vector search service (T024)
   - Query preprocessing → preprocessing service (T023)
   - Hybrid ranking → ranking service (T025)

## Validation Checklist

### Validation Gate

GATE: Checked by main() before returning

- [x] All contracts have corresponding tests (T005-T007)
- [x] All entities have model tasks (T018-T021)
- [x] All tests come before implementation (Phase 3.2 before 3.3)
- [x] Parallel tasks truly independent (different files, no shared state)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Constitutional requirements addressed (MCP-first, performance, citations)
- [x] Integration with existing KCS architecture planned
