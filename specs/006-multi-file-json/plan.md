# Implementation Plan: Multi-File JSON Output Strategy

**Branch**: `006-multi-file-json` | **Date**: 2025-01-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-multi-file-json/spec.md`

## Execution Flow (/plan command scope)

```text
1. Load feature spec from Input path
   → Successfully loaded spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Using existing KCS tech stack
   → Structure Decision: Single project (Rust + Python + SQL)
3. Evaluate Constitution Check section below
   → No violations - approach aligns with principles
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → Researching chunking strategies and memory management
5. Execute Phase 1 → contracts, data-model.md, quickstart.md
   → Creating MCP contracts and data models for chunked processing
6. Re-evaluate Constitution Check section
   → No new violations introduced
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach
8. STOP - Ready for /tasks command
```

## Summary

Implement multi-file JSON output strategy to replace the current single 2.8GB JSON file with multiple 50MB chunks organized by kernel subsystem. This enables parallel processing, resumable operations, and memory-efficient database population while maintaining the existing Rust parser → Python MCP server → PostgreSQL pipeline.

## Technical Context

**Language/Version**: Rust 1.75+ (parsers), Python 3.11+ (MCP server)
**Primary Dependencies**:
  - Rust: tree-sitter, clang, rayon, serde_json, PyO3
  - Python: FastAPI, asyncpg, structlog, pydantic
  - Database: PostgreSQL 14+ with pgvector extension
**Storage**: PostgreSQL for parsed data, filesystem for JSON chunks
**Testing**: cargo test (Rust), pytest (Python), k6 (performance)
**Target Platform**: Linux server (Ubuntu 22.04+)
**Project Type**: single - Multi-language pipeline (Rust + Python + SQL)
**Performance Goals**:
  - Full kernel index ≤30 minutes (increased from 20min due to chunking overhead)
  - Chunk processing p95 ≤600ms per chunk
  - Memory usage ≤250MB per chunk (5x chunk size)
**Constraints**:
  - Read-only kernel access
  - Backward compatibility with existing MCP endpoints
  - Must support incremental updates
**Scale/Scope**:
  - ~70,000 kernel source files
  - ~60 chunks @ 50MB each
  - Support parallel processing of 4-8 chunks

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (existing KCS project with multi-language components)
- Using framework directly? Yes - direct use of serde_json, asyncpg
- Single data model? Yes - extending existing Symbol/EntryPoint models
- Avoiding patterns? Yes - no new patterns, using existing streaming approach

**Architecture**:
- EVERY feature as library? Yes - chunking in kcs-serializer library
- Libraries listed:
  - kcs-serializer: JSON chunking and manifest generation
  - kcs-python-bridge: Chunk processing coordination
- CLI per library:
  - kcs-parser --chunk-size --output-pattern
  - kcs-extractor --input-manifest
- Library docs: Will update existing llms.txt with chunk options

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes - real PostgreSQL, real files
- Integration tests for: chunk boundaries, manifest validation, resume logic
- FORBIDDEN: No implementation before tests

**Observability**:
- Structured logging included? Yes - structlog for chunk progress
- Frontend logs → backend? N/A (CLI tool)
- Error context sufficient? Yes - chunk ID, file path, error details

**Versioning**:
- Version number assigned? Using existing KCS versioning
- BUILD increments on every change? Yes
- Breaking changes handled? Backward compatible - single file mode retained

## Project Structure

### Documentation (this feature)

```text
specs/006-multi-file-json/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

```text
# Existing KCS structure - Single project with multi-language components
src/
├── rust/
│   ├── kcs-parser/       # Add chunking flags
│   ├── kcs-serializer/   # New chunk writer implementation
│   └── kcs-python-bridge/# Add chunk coordination
├── python/
│   └── kcs_mcp/
│       ├── chunk_loader.py  # New chunk processing module
│       └── tools.py         # Update for chunk-aware queries
└── sql/
    └── migrations/
        └── 012_chunk_tracking.sql  # Track processed chunks

tests/
├── contract/            # Chunk manifest schema tests
├── integration/         # Multi-chunk processing tests
├── performance/         # Chunk size optimization tests
└── fixtures/
    └── chunks/         # Sample chunk files for testing

tools/
├── index_kernel.sh     # Update with chunking flags
└── process_chunks.py   # New chunk processor script
```

**Structure Decision**: Option 1 (Single project) - Extending existing KCS structure

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - Optimal chunk size vs processing overhead tradeoff
   - Checksum algorithm for chunk validation (MD5 vs SHA256)
   - Chunk naming strategy for subsystem organization
   - Parallel processing coordination mechanism

2. **Generate and dispatch research agents**:
   ```
   Task: "Research JSON streaming chunking strategies in Rust"
   Task: "Find best practices for resumable batch processing"
   Task: "Research PostgreSQL transaction boundaries for bulk inserts"
   Task: "Evaluate memory-mapped files vs streaming for large JSON"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: Chunk size fixed at 50MB
   - Rationale: Balance between memory usage and file count
   - Alternatives considered: 10MB (too many files), 100MB (memory issues)

**Output**: research.md with all technical decisions resolved

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - ChunkManifest: metadata about all chunks
   - ChunkFile: individual chunk with symbols/entries
   - ProcessingStatus: tracks completion state
   - ChunkMetadata: size, checksum, subsystem, file count

2. **Generate API contracts** from functional requirements:
   - GET /mcp/chunks/manifest - retrieve chunk listing
   - GET /mcp/chunks/{id}/status - check processing state
   - POST /mcp/chunks/{id}/process - trigger chunk load
   - Output OpenAPI schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - test_manifest_schema.py - validate manifest structure
   - test_chunk_status.py - verify status endpoints
   - test_chunk_processing.py - test load operations
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Full indexing with 60 chunks scenario
   - Resume from chunk 15 failure scenario
   - Incremental update of network subsystem scenario

5. **Update CLAUDE.md incrementally**:
   - Add chunking flags to index_kernel.sh examples
   - Document new chunk processing workflow
   - Add troubleshooting for chunk failures

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md updates

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs:
  - Contract tests for manifest and status endpoints [P]
  - Rust kcs-serializer chunk writer implementation
  - Python chunk_loader module with resume logic
  - Database migration for chunk tracking table
  - Integration tests for multi-chunk processing
  - Update index_kernel.sh with chunking flags

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Rust serializer → Python loader → Shell script
- Mark [P] for parallel execution where possible

**Estimated Output**: 20-25 numbered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following TDD)
**Phase 5**: Validation (run integration tests, performance benchmarks)

## Complexity Tracking

*No violations - using existing patterns and architecture*

## Progress Tracking

*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - approach described)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*