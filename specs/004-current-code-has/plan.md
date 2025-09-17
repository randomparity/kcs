# Implementation Plan: Enhanced Kernel Entry Point and Symbol Detection

**Branch**: `004-current-code-has` | **Date**: 2025-09-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-current-code-has/spec.md`

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

Enhance KCS to detect and analyze all major kernel entry points (ioctl, file_ops, sysfs, procfs, debugfs, netlink) beyond just syscalls, implement kernel-specific pattern recognition (EXPORT_SYMBOL, module_param), and enrich symbols with semantic analysis from Clang. This will provide comprehensive kernel interface discovery and accurate dependency tracking for kernel developers and AI assistants.

## Technical Context

**Language/Version**: Rust 1.75 (parser/extractor), Python 3.11 (MCP server)
**Primary Dependencies**: tree-sitter, clang-sys, sqlx (Rust); FastAPI, asyncpg (Python)
**Storage**: PostgreSQL 15+ with pgvector extension
**Testing**: cargo test (Rust), pytest (Python), k6 (performance)
**Target Platform**: Linux server (Ubuntu 22.04+)
**Project Type**: single - existing KCS project with established structure
**Performance Goals**: Pattern detection <100ms per file, extraction <20min for full kernel
**Constraints**: Query p95 <600ms (constitutional), memory <4GB during indexing
**Scale/Scope**: ~50k symbols, ~500k edges, 10+ entry point types, 6 kernel configs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 2 (Rust components, Python MCP server)
- Using framework directly? Yes - tree-sitter and clang APIs directly
- Single data model? Yes - existing database schema extended
- Avoiding patterns? Yes - direct pattern matching, no abstraction layers

**Architecture**:

- EVERY feature as library? Yes - kcs-extractor library enhanced
- Libraries listed:
  - kcs-extractor: Entry point detection (enhanced)
  - kcs-parser: Symbol enrichment with Clang (enhanced)
- CLI per library: extract-entry-points CLI already exists
- Library docs: Rust docs + README per crate

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor cycle enforced? Yes - tests first
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes - real kernel repos, actual Postgres
- Integration tests for: new entry points, pattern detection, Clang integration? Yes
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:

- Structured logging included? Yes - existing tracing setup
- Frontend logs → backend? N/A (no frontend)
- Error context sufficient? Yes - file:line citations always included

**Versioning**:

- Version number assigned? Continue existing 0.2.0
- BUILD increments on every change? Via CI/CD
- Breaking changes handled? No schema changes, backward compatible

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

**Structure Decision**: Use existing KCS project structure (Rust workspace + Python package)

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
- Tasks will be organized in phases:
  1. Database schema migration (1 task)
  2. Contract tests for 3 new endpoints (3 tasks [P])
  3. Pattern detection in Rust (5-6 tasks for different patterns)
  4. Entry point extraction enhancement (4-5 tasks for new types)
  5. Clang integration implementation (2-3 tasks)
  6. Python integration layer updates (2-3 tasks)
  7. Integration tests from quickstart scenarios (5 tasks)
  8. Performance validation (1 task)

**Ordering Strategy**:

- TDD order: Tests before implementation
- Dependency order:
  1. Schema migration first
  2. Contract tests (can be parallel)
  3. Rust implementation (pattern detection → entry points → Clang)
  4. Python integration
  5. Integration tests
  6. Performance validation
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**Key Implementation Areas**:

1. **Pattern Detection** (kernel_patterns.rs):
   - EXPORT_SYMBOL variants
   - module_param variants
   - Setup/early/core params

2. **Entry Point Types** (entry_points.rs):
   - Enhance existing ioctl detection
   - Add procfs/debugfs patterns
   - Add netlink/notification chains
   - Add interrupt handlers

3. **Clang Bridge** (clang_bridge.rs):
   - Initialize Clang index
   - Parse compile_commands.json
   - Extract type information
   - Enhance symbols with semantic data

4. **Database Layer**:
   - Migration for metadata columns
   - New kernel_pattern table
   - Update insertion methods

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
- [x] Complexity deviations documented (none needed)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
