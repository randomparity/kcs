# Implementation Plan: Infrastructure Core Components Implementation

**Branch**: `005-infrastructure-empty-stub` | **Date**: 2025-09-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-infrastructure-empty-stub/spec.md`

## Execution Flow (/plan command scope)

```text
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md,
   agent-specific template file (e.g., `CLAUDE.md` for Claude Code,
   `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md`
   for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```text

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are
executed by other commands:

- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

This feature implements critical infrastructure components for KCS that are currently stubbed or missing: kernel configuration parsing for multi-architecture support, drift detection for spec vs implementation validation, semantic search capabilities, call graph traversal with cycle detection, and graph serialization for analysis export. These components address fundamental limitations affecting config-aware analysis, query capabilities, and validation features.

## Technical Context

**Language/Version**: Rust 1.75+, Python 3.11+
**Primary Dependencies**: tree-sitter, clang-sys, PyO3, FastAPI, PostgreSQL, pgvector
**Storage**: PostgreSQL with pgvector extension, JSONB metadata columns
**Testing**: pytest, cargo test, k6 performance tests
**Target Platform**: Linux server (Ubuntu 22.04+)
**Project Type**: single - KCS is a monorepo with Rust components and Python MCP server
**Performance Goals**: Query p95 ≤600ms, full index ≤20min, incremental ≤3min (from constitution)
**Constraints**: Handle 50,000+ input files on 16GB RAM systems, output files parseable by jq (from FR-009)
**Scale/Scope**: Linux kernel codebase (20M+ LOC), multiple architectures (x86, ARM64, RISC-V, PowerPC)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 1 (KCS monorepo with Rust crates + Python MCP server)
- Using framework directly? YES (FastAPI, tree-sitter, clang-sys without wrappers)
- Single data model? YES (PostgreSQL schema, no DTOs)
- Avoiding patterns? YES (direct DB queries, no Repository pattern)

**Architecture**:

- EVERY feature as library? YES (all components are crates/modules)
- Libraries listed:
  - kcs-config: Kernel configuration parsing (new)
  - kcs-drift: Spec vs implementation validation (new)
  - kcs-search: Semantic search with pgvector (new)
  - kcs-graph (enhanced): Call graph traversal with cycle detection
  - kcs-serializer: Graph export to JSON/GraphML (new)
- CLI per library: Each crate has CLI with --help/--version/--format
- Library docs: llms.txt format planned? YES (per crate)

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? YES
- Git commits show tests before implementation? YES
- Order: Contract→Integration→E2E→Unit strictly followed? YES
- Real dependencies used? YES (PostgreSQL with pgvector)
- Integration tests for: new libraries, contract changes, shared schemas? YES
- FORBIDDEN: Implementation before test, skipping RED phase - UNDERSTOOD

**Observability**:

- Structured logging included? YES (structlog for Python, env_logger for Rust)
- Frontend logs → backend? N/A (no frontend)
- Error context sufficient? YES (file:line citations required)

**Versioning**:

- Version number assigned? Using existing KCS version scheme
- BUILD increments on every change? YES
- Breaking changes handled? YES (migration scripts for DB schema)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT
                         # created by /plan)
```text

### Source Code (repository root)

```text
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```text

**Structure Decision**: Option 1 (Single project) - KCS monorepo structure

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```

   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"

   ```text

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved ✓ COMPLETE

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/bash/update-agent-context.sh claude` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md ✓, /contracts/* ✓, failing tests ✓, quickstart.md ✓,
agent-specific file (CLAUDE.md updated) ✓ COMPLETE

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute
during /plan*

**Task Generation Strategy**:

The /tasks command will generate approximately 40-45 tasks organized as follows:

1. **Database Migration Tasks** (5 tasks):
   - Create migration for kernel_config table [P]
   - Create migration for specification table [P]
   - Create migration for drift_report table [P]
   - Create migration for semantic_query_log table [P]
   - Create migration for graph_export table [P]

2. **Contract Test Tasks** (5 tasks):
   - Write failing test for parse_kernel_config endpoint
   - Write failing test for validate_spec endpoint
   - Write failing test for semantic_search endpoint
   - Write failing test for traverse_call_graph endpoint
   - Write failing test for export_graph endpoint

3. **Rust Crate Implementation Tasks** (20 tasks):
   - kcs-config crate (4 tasks): structure, parser, CLI, tests
   - kcs-drift completion (4 tasks): drift_detector, report_generator, integration, tests
   - kcs-search crate (4 tasks): embeddings, query, pgvector integration, tests
   - kcs-graph enhancements (4 tasks): cycle detection, path reconstruction, traversal, tests
   - kcs-serializer crate (4 tasks): JSON export, GraphML export, chunking, tests

4. **Python MCP Integration Tasks** (10 tasks):
   - Implement parse_kernel_config endpoint
   - Implement validate_spec endpoint
   - Implement semantic_search endpoint
   - Implement traverse_call_graph endpoint
   - Implement export_graph endpoint
   - Database query implementations (5 tasks)

5. **Integration & Performance Tasks** (5 tasks):
   - End-to-end multi-arch config test
   - Semantic search accuracy test
   - Graph traversal performance test
   - Large graph export memory test
   - Full quickstart validation

**Ordering Strategy**:

- Phase 1: Database migrations (parallel)
- Phase 2: Contract tests (must fail first)
- Phase 3: Rust crate implementations (some parallel)
- Phase 4: Python MCP endpoints
- Phase 5: Integration tests
- Mark [P] for parallel execution where no dependencies exist

**TDD Enforcement**:
- Every implementation task must have a corresponding failing test first
- Tests must be committed before implementation
- Integration tests before feature integration

**Estimated Output**: 40-45 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional
principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance
validation)

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access |
|                           |                    | insufficient]        |

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
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
