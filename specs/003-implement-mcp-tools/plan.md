# Implementation Plan: Implement MCP Tools

**Branch**: `003-implement-mcp-tools` | **Date**: 2025-09-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-implement-mcp-tools/spec.md`

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

Replace mock implementations in MCP tool endpoints with actual database-backed functionality. The system needs to query and traverse call graph data stored in PostgreSQL to provide accurate kernel analysis for four key endpoints: who_calls, list_dependencies, entrypoint_flow, and impact_of. All responses must include proper file:line citations from the actual kernel source.

## Technical Context

**Language/Version**: Python 3.11 (MCP server), Rust 1.75 (parser)
**Primary Dependencies**: FastAPI, asyncpg, pydantic, structlog
**Storage**: PostgreSQL 15+ with pgvector extension
**Testing**: pytest with pytest-asyncio, cargo test for Rust
**Target Platform**: Linux server hosting MCP API
**Project Type**: single (existing KCS project structure)
**Performance Goals**: Query p95 < 600ms (constitutional requirement)
**Constraints**: Read-only on kernel source, all results must include citations
**Scale/Scope**: ~50k kernel symbols, ~10k entry points, 6 kernel configs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 2 (Python MCP server, Rust parser/extractor)
- Using framework directly? Yes (FastAPI, asyncpg directly)
- Single data model? Yes (Pydantic models match DB schema)
- Avoiding patterns? Yes (direct DB queries, no abstraction layers)

**Architecture**:

- EVERY feature as library? Yes (kcs_mcp package)
- Libraries listed:
  - kcs_mcp: MCP protocol server implementation
  - kcs_parser: Rust parser with Python bindings
- CLI per library: kcs-mcp server CLI with --host/--port/--help
- Library docs: MCP protocol self-documenting via resources endpoint

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? Yes
- Git commits show tests before implementation? Will enforce
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (real PostgreSQL via testcontainers)
- Integration tests for: new libraries, contract changes, shared schemas? Yes
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:

- Structured logging included? Yes (structlog)
- Frontend logs → backend? N/A (API only)
- Error context sufficient? Yes (request IDs, symbol context)

**Versioning**:

- Version number assigned? 1.0.0 (from pyproject.toml)
- BUILD increments on every change? Via CI/CD
- Breaking changes handled? N/A (replacing mocks, not breaking API)

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

**Structure Decision**: Existing KCS project structure (src/python/, src/rust/, tests/)

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

**Output**: research.md with all NEEDS CLARIFICATION resolved

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

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md,
agent-specific file

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute
during /plan*

**Task Generation Strategy**:

- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Contract tests for each endpoint (4 tasks) [P]
- Database method enhancements (2 tasks)
- Implementation for each endpoint (4 tasks)
- Integration tests from quickstart scenarios (4 tasks)
- Performance validation task

**Ordering Strategy**:

1. Contract tests first (TDD - must fail initially)
2. Database enhancements (depth traversal, cycle detection)
3. Endpoint implementations (make contract tests pass)
4. Integration tests (validate end-to-end flows)
5. Performance benchmarks (verify < 600ms requirement)

**Task Categories**:

- **Testing Tasks** (8 total):
  - 4 contract tests (one per endpoint)
  - 4 integration tests (from quickstart scenarios)

- **Implementation Tasks** (6 total):
  - 2 database method enhancements
  - 4 endpoint implementations (remove mocks)

- **Validation Tasks** (1 total):
  - Performance benchmark verification

**Estimated Output**: 15 numbered, ordered tasks in tasks.md

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
