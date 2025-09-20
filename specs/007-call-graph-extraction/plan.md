# Implementation Plan: Call Graph Extraction Specification

**Branch**: `007-call-graph-extraction` | **Date**: 2025-01-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/007-call-graph-extraction/spec.md`

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

Call graph extraction specification for comprehensive analysis of function call relationships in Linux kernel code. The primary requirement is to extract and classify all types of function calls (direct, indirect, macro-based, callback) from C source code using AST traversal, providing complete relationship data for MCP protocol endpoints. The system must handle large kernel codebases (50,000+ files) within 20-minute performance constraints while integrating with existing Tree-sitter infrastructure.

## Technical Context

**Language/Version**: Rust 1.75+ (primary implementation), Python 3.11+ (MCP server integration)
**Primary Dependencies**: Tree-sitter (AST parsing), petgraph (graph algorithms), clang-sys (optional semantic analysis), rayon (parallel processing)
**Storage**: PostgreSQL 15+ with pgvector extension (call graph storage), newline-delimited JSON (streaming output)
**Testing**: cargo test (Rust components), pytest (Python integration), real kernel fixtures
**Target Platform**: Linux development servers, container environments
**Project Type**: single (existing KCS codebase extension)
**Performance Goals**: 20-minute full kernel indexing, sub-second MCP query responses, 90%+ call edge accuracy
**Constraints**: Read-only operation, constitutional compliance, Tree-sitter integration, MCP protocol compatibility
**Scale/Scope**: 50,000+ kernel source files, millions of function calls, multiple kernel configurations

**Additional Context**: Project tooling well documented at this stage.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**KCS Constitutional Requirements**:

- **Read-Only Safety**: ✅ Call graph extraction operates in read-only mode on kernel repositories
- **Citation-Based Truth**: ✅ All call relationships include exact file/line citations
- **MCP-First Interface**: ✅ Functionality exposed through MCP protocol endpoints
- **Configuration Awareness**: ✅ Call graphs tagged with kernel configuration context
- **Performance Boundaries**: ✅ Meets 20-minute indexing, 600ms query p95 targets

**Simplicity**:

- Projects: 1 (kcs-parser extension within existing codebase)
- Using framework directly? ✅ (Tree-sitter, petgraph, no wrappers)
- Single data model? ✅ (CallEdge, Function entities match requirements)
- Avoiding patterns? ✅ (Direct graph algorithms, no unnecessary abstractions)

**Architecture**:

- EVERY feature as library? ✅ (kcs-parser crate provides call graph extraction)
- Libraries listed: kcs-parser (AST traversal + call extraction), kcs-graph (relationship storage)
- CLI per library: ✅ (kcs-parser --include-calls, --help, --version)
- Library docs: ✅ (CLAUDE.md format exists and maintained)

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? ✅ (Contract tests before implementation)
- Git commits show tests before implementation? ✅ (TDD process documented)
- Order: Contract→Integration→E2E→Unit strictly followed? ✅ (KCS testing approach)
- Real dependencies used? ✅ (actual PostgreSQL, real kernel fixtures)
- Integration tests for: ✅ (MCP endpoints, graph algorithms, Tree-sitter integration)
- FORBIDDEN: Implementation before test, skipping RED phase ✅

**Observability**:

- Structured logging included? ✅ (structlog for Python, tracing for Rust)
- Unified logging stream? ✅ (consistent format across components)
- Error context sufficient? ✅ (file citations, parse errors, graph validation)

**Versioning**:

- Version number assigned? ✅ (1.0.0 in pyproject.toml, follows semver)
- BUILD increments on every change? ✅ (CI/CD processes)
- Breaking changes handled? ✅ (constitutional compliance for API stability)

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

**Structure Decision**: [DEFAULT to Option 1 unless Technical Context
indicates web/mobile app]

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

- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- API contract endpoints → contract test tasks [P] (4 endpoints)
- Data model entities → Rust struct/table creation tasks [P] (6 entities)
- Quickstart scenarios → integration test tasks (4 scenarios)
- Tree-sitter query implementation → AST pattern matching tasks
- MCP endpoint implementation → service layer tasks
- Database migration → schema update tasks

**Ordering Strategy**:

- TDD order: Tests before implementation (contract tests first)
- Dependency order: Data models → AST processing → MCP endpoints → integration
- Tree-sitter patterns before call extraction logic
- Database schema before service implementation
- Mark [P] for parallel execution (independent modules)

**Estimated Task Breakdown**:
1. Contract tests (4 tasks) - [P]
2. Data model creation (6 tasks) - [P]
3. Tree-sitter query patterns (5 tasks) - sequential
4. AST call extraction (8 tasks) - sequential
5. Database integration (4 tasks) - [P] after schema
6. MCP endpoint implementation (4 tasks) - after extraction
7. Integration tests (4 tasks) - after endpoints
8. Performance optimization (3 tasks) - final

**Estimated Output**: 38 numbered, ordered tasks in tasks.md

**Key Dependencies Identified**:
- Tree-sitter queries must be completed before call extraction
- Database schema changes before service implementation
- Call extraction before MCP endpoint integration
- Contract tests can run in parallel from start

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
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
