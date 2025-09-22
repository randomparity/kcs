# Implementation Plan: Semantic Search Engine (kcs-search)

**Branch**: `008-semantic-search-engine` | **Date**: 2025-09-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/dave/src/kcs/specs/008-semantic-search-engine/spec.md`

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

Implement a semantic search engine that allows kernel researchers and security analysts to search large kernel codebases using natural language queries. The system will use the BAAI/bge-small-en-v1.5 embedding model for CPU-only operation, pgvector for efficient vector storage and retrieval, and expose functionality through MCP endpoints for integration with analysis tools.

## Technical Context

**Language/Version**: Python 3.11+ (for existing KCS ecosystem compatibility)
**Primary Dependencies**: sentence-transformers (BAAI/bge-small-en-v1.5), pgvector, PostgreSQL, FastAPI
**Storage**: PostgreSQL with pgvector extension for vector embeddings and metadata
**Testing**: pytest (following existing KCS patterns)
**Target Platform**: Linux server (CPU-only operation)
**Project Type**: single (library integration into existing KCS system)
**Performance Goals**: Query response p95 ≤600ms (per constitution), indexing 10k files/hour
**Constraints**: CPU-only operation, read-only on kernel source, MCP interface required
**Scale/Scope**: Large kernel codebases (millions of LOC), 10+ concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 1 (integrating into existing KCS system)
- Using framework directly? Yes (FastAPI for MCP endpoints, sentence-transformers directly)
- Single data model? Yes (SearchResult, VectorEmbedding entities)
- Avoiding patterns? Yes (direct database access, no Repository pattern unless needed)

**Architecture**:

- EVERY feature as library? Yes (kcs-search as library with CLI interface)
- Libraries listed: kcs-search (semantic search with embedding generation and vector storage)
- CLI per library: kcs-search with --help/--version/--format, search subcommands
- Library docs: llms.txt format planned? Yes

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? Yes (failing tests first)
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (actual PostgreSQL with pgvector)
- Integration tests for: MCP endpoints, embedding generation, vector search accuracy
- FORBIDDEN: Implementation before test, skipping RED phase

**Observability**:

- Structured logging included? Yes (following KCS logging patterns)
- Frontend logs → backend? N/A (server-side only)
- Error context sufficient? Yes (with file citations per constitution)

**Versioning**:

- Version number assigned? 0.1.0 (semantic search MVP)
- BUILD increments on every change? Yes
- Breaking changes handled? Yes (parallel tests for MCP interface changes)

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

**Structure Decision**: Option 1 (Single project) - integrating into existing KCS architecture

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
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:

- TDD order: Tests before implementation
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

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
- [x] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
