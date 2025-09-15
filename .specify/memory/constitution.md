# KCS Constitution

## Core Principles

### I. Read-Only Safety

The KCS system must operate in read-only mode on kernel repositories. No code generation, modification, or mutation of kernel source is permitted. All operations must be non-destructive and auditable.

### II. Citation-Based Truth

Every claim, finding, or recommendation must include exact file/line citations. No unsourced assertions are permitted. All analysis results must be traceable to specific kernel source locations.

### III. MCP-First Interface

All functionality must be exposed through Model Context Protocol (MCP) resources and tools. The MCP interface is the primary contract with AI agents and developers. Text-based queries in, structured JSON with citations out.

### IV. Configuration Awareness

All analysis must be configuration-aware. Symbol presence, call graphs, and dependencies vary by kernel configuration (defconfig, allmodconfig, etc). Results must clearly indicate their configuration context.

### V. Performance Boundaries

System must meet defined performance targets: full index ≤20min, incremental ≤3min, query p95 ≤600ms. Performance degradation blocks deployment. Cache aggressively but invalidate correctly.

## Quality Requirements

### Static Analysis Coverage

- Minimum 90% first-hop edge resolution for modified files
- Minimum 95% ioctl decoding accuracy
- All entry points must be identified and typed

### Testing Standards

- Unit tests for all graph algorithms
- Integration tests for MCP endpoints
- Regression tests for known kernel patterns
- Performance benchmarks with alerts on degradation

## Security & Privacy

### Access Control

- Token-based authentication required
- Per-project access scoping
- Read-only enforcement at all layers

### Data Protection

- No kernel source in logs
- PII redaction in all outputs
- Audit trail for all queries

## Governance

The constitution supersedes all implementation decisions. Changes require:

1. Documentation of rationale
2. Impact analysis on existing functionality
3. Migration plan for breaking changes

All code reviews must verify constitutional compliance. Violations block merge.

**Version**: 1.0.0 | **Ratified**: 2025-09-14 | **Last Amended**: 2025-09-14
