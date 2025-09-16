# Feature Specification: Call Graph Extraction

**Feature Branch**: `002-1-call-graph`
**Created**: 2025-09-16
**Status**: Draft
**Input**: User description: "1. Call Graph Extraction

- Location: src/rust/kcs-parser/src/lib.rs:265
- Issue: TODO: Extract call edges from AST - parser returns empty call_edges
- Impact: Breaks who_calls, list_dependencies, entrypoint_flow, impact_of MCP endpoints
- Priority: Highest - Core feature missing"

## Execution Flow (main)

```text
1. Parse user description from Input
   � Feature identified: Call graph extraction for kernel code analysis
2. Extract key concepts from description
   � Actors: Kernel analysts, AI coding assistants
   � Actions: Extract function call relationships from C code
   � Data: Call edges between functions in kernel source
   � Constraints: Must work with Tree-sitter AST parsing
3. For each unclear aspect:
   � All core aspects clear from technical context provided
4. Fill User Scenarios & Testing section
   � User flow: Query function relationships via MCP endpoints
5. Generate Functional Requirements
   � Requirements focus on call graph data extraction and API availability
6. Identify Key Entities
   � CallEdge entities representing function call relationships
7. Run Review Checklist
   � Spec focused on user needs rather than implementation
8. Return: SUCCESS (spec ready for planning)
```

---

## � Quick Guidelines

-  Focus on WHAT users need and WHY
- L Avoid HOW to implement (no tech stack, APIs, code structure)
- =e Written for business stakeholders, not developers

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story

As a kernel developer or AI coding assistant analyzing Linux kernel code, I need to understand function call relationships so that I can trace code paths, analyze impact of changes, and understand system dependencies.

### Acceptance Scenarios

1. **Given** a kernel source file with function calls, **When** I query "who calls function X", **Then** I receive a list of all functions that call X with file locations
2. **Given** a function name, **When** I query its dependencies, **Then** I receive all functions it calls directly
3. **Given** an entry point (syscall, ioctl, etc.), **When** I trace its flow, **Then** I can follow the complete call chain from entry to implementation
4. **Given** a function I want to modify, **When** I analyze its impact, **Then** I can see all code paths that would be affected by the change

### Edge Cases

- What happens when function calls are made through function pointers?
- How does the system handle indirect calls and callbacks?
- What about calls through macros or preprocessor definitions?
- How are calls handled in conditional compilation blocks?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract direct function call relationships from C source code
- **FR-002**: System MUST provide call graph data via MCP endpoints (who_calls, list_dependencies, entrypoint_flow, impact_of)
- **FR-003**: Users MUST be able to query which functions call a specific target function
- **FR-004**: Users MUST be able to query which functions a specific source function calls
- **FR-005**: System MUST trace complete call paths from kernel entry points to implementation functions
- **FR-006**: System MUST analyze the blast radius of changes by following call relationships
- **FR-007**: Call relationships MUST include file location citations (file:line references)
- **FR-008**: System MUST handle C-specific constructs like function pointers and callbacks [NEEDS CLARIFICATION: level of indirect call support required]
- **FR-009**: System MUST integrate with existing Tree-sitter AST parsing without breaking current symbol extraction
- **FR-010**: System MUST provide call graph data with kernel configuration awareness

### Key Entities *(include if feature involves data)*

- **CallEdge**: Represents a function call relationship with caller, callee, call site location, and optional context (conditional compilation, call type)
- **Function**: Represents callable entities in kernel code with name, signature, location, and call relationships
- **CallSite**: Represents the specific location where a function call occurs, including file path and line number

---

## Review & Acceptance Checklist

*GATE: Automated checks run during main() execution*

### Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness

- [ ] No [NEEDS CLARIFICATION] markers remain (FR-008 needs clarification on indirect call support)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarification)

---
