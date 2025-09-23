# Implementation Plan: Call Graph Extraction

**Branch**: `002-1-call-graph` | **Date**: 2025-09-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/dave/src/kcs/specs/002-1-call-graph/spec.md`

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

Implement call graph extraction in the KCS parser to identify function call relationships from C source code using Tree-sitter AST. This enables MCP endpoints (who_calls, list_dependencies, entrypoint_flow, impact_of) to provide kernel code analysis capabilities for AI assistants and developers.

## Technical Context

**Language/Version**: Rust 1.75, Python 3.11
**Primary Dependencies**: tree-sitter, tree-sitter-c, clang-sys, tokio, serde
**Storage**: PostgreSQL 15+ with pgvector extension
**Testing**: cargo test, pytest with real PostgreSQL database
**Target Platform**: Linux server (MCP protocol server)
**Project Type**: single (multi-language Rust workspace + Python package)
**Performance Goals**: Parse kernel files at >10k lines/sec, call graph extraction <5ms per function
**Constraints**: Must preserve existing symbol extraction, <600ms p95 query response, read-only kernel source
**Scale/Scope**: Linux kernel (~30M LOC), ~50k symbols, ~10k entry points per configuration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 2 (kcs-parser Rust library, tests) - PASS
- Using framework directly? Yes, tree-sitter and tokio directly - PASS
- Single data model? Yes, CallEdge struct matches database model - PASS
- Avoiding patterns? Direct AST traversal, no complex abstractions - PASS

**Architecture**:

- EVERY feature as library? Yes, kcs-parser provides lib + CLI - PASS
- Libraries listed: kcs-parser (call graph extraction from C source)
- CLI per library: kcs-parser parse --help/--version - PASS
- Library docs: Will update existing llms.txt format - PASS

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? Yes, will create failing tests first - PASS
- Git commits show tests before implementation? Yes, constitutional requirement - PASS
- Order: Contract→Integration→E2E→Unit strictly followed? Yes - PASS
- Real dependencies used? Yes, actual tree-sitter, real test kernel files - PASS
- Integration tests for: call graph extraction, MCP endpoint changes - PASS
- FORBIDDEN: Implementation before test, skipping RED phase - UNDERSTOOD

**Observability**:

- Structured logging included? Yes, using tracing crate - PASS
- Frontend logs → backend? N/A, server-side only - PASS
- Error context sufficient? Yes, detailed parse error reporting - PASS

**Versioning**:

- Version number assigned? 1.0.0 (existing) - PASS
- BUILD increments on every change? Yes, following semantic versioning - PASS
- Breaking changes handled? None expected, additive feature - PASS

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

**Structure Decision**: Option 1 (Single project) - KCS is a specialized kernel analysis system with Rust performance-critical components and Python MCP server

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

- Load `/templates/tasks-template.md` as base template
- Generate tasks from Phase 1 artifacts:
  - data-model.md → CallEdge struct implementation [P]
  - contracts/parser-api.json → contract test for parse_file endpoint [P]
  - contracts/mcp-endpoints.json → contract tests for 4 MCP endpoints [P]
  - quickstart.md → 5 integration test scenarios
- Tree-sitter integration tasks for AST traversal
- Database integration tasks for call edge persistence
- Performance benchmark tasks

**Ordering Strategy**:

- Contract tests first (RED phase - must fail initially)
- Data model implementation (CallEdge struct, ParseResult extension)
- Core parser logic (tree-sitter AST traversal for calls)
- Database integration (call edge persistence)
- MCP endpoint updates to use call graph data
- Integration tests (quickstart scenarios)
- Performance optimization and benchmarking
- Mark [P] for parallel execution where dependencies allow

**Estimated Output**: 20-25 numbered, ordered tasks focusing on:
- 5 contract test tasks
- 3 data model tasks
- 4 parser implementation tasks
- 3 database integration tasks
- 4 MCP endpoint tasks
- 5 integration test tasks
- 2 performance tasks

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
- [x] Complexity deviations documented (None - all constitutional checks passed)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
