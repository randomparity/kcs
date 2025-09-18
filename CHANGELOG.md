# Changelog

All notable changes to the Kernel Context Server (KCS) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### MCP Tools Implementation (Phase 3)

- **Database Enhancements**
  - Recursive CTE queries for proper depth-based call graph traversal
  - Cycle detection using visited path array tracking to prevent infinite loops
  - Maximum depth enforcement (5 levels) to prevent excessive traversal
  - Performance optimizations with LIMIT 100 for all graph queries

- **Enhanced MCP Endpoints**
  - `who_calls`: Now uses real database queries with depth traversal (1-5 levels)
  - `list_dependencies`: Now uses real database queries with depth traversal (1-5 levels)
  - `entrypoint_flow`: Extended support for 40+ syscalls, ioctl commands, and file_ops entry points
  - `impact_of`: Enhanced symbol extraction, bidirectional traversal, subsystem detection

- **Comprehensive Test Suite**
  - Contract tests for all 4 MCP endpoints validating request/response schemas
  - Integration tests for citations, depth, cycles, and empty results
  - Performance validation tests ensuring p95 < 600ms constitutional requirement
  - Production-scale test data generation (50k symbols, 500k edges)

### Changed

- Removed all mock data fallbacks from MCP endpoints - now using only real database queries
- Increased `entrypoint_flow` max depth from 3 to 5 for better flow tracing
- Enhanced risk calculation in `impact_of` based on blast radius size

### Fixed

- Cycle detection in call graph traversal preventing infinite loops
- Proper handling of empty results across all endpoints
- Depth parameter now properly respected in traversal queries

### Performance

- All MCP endpoints meet constitutional requirement of p95 < 600ms
- Concurrent request handling tested with 20 parallel requests
- Worst-case scenarios (depth=5, large blast radius) still meet performance targets

#### Infrastructure Core Components (Phase 5)

- **New Infrastructure Rust Crates**
  - `kcs-config`: Kernel configuration file parsing with dependency resolution
  - `kcs-drift`: Specification validation and compliance scoring with drift detection
  - `kcs-search`: Semantic search using vector embeddings and pgvector integration
  - `kcs-graph`: Advanced call graph traversal with cycle detection and path finding
  - `kcs-serializer`: Multi-format graph export (JSON, GraphML, DOT, CSV) with compression

- **New MCP Infrastructure Endpoints**
  - `parse_kernel_config`: Parse .config files with architecture and dependency support
  - `validate_spec`: Validate specifications against kernel implementation with compliance scoring
  - `semantic_search`: AI-powered semantic code search with embeddings and explanations
  - `traverse_call_graph`: Advanced graph traversal with cycle detection and visualization
  - `export_graph`: Export call graphs in multiple formats with chunking and compression

- **Database Schema Enhancements**
  - New tables: `kernel_config`, `specification`, `drift_report`, `semantic_query_log`, `graph_export`
  - Config awareness columns added to existing tables for multi-configuration support
  - pgvector integration for semantic search capabilities

- **Advanced Features**
  - Configuration-aware analysis supporting multiple kernel configs
    (x86_64:defconfig, arm64:defconfig, etc.)
  - Specification validation with compliance scoring and drift detection
  - Semantic search with query expansion, reranking, and result explanations
  - Graph export with compression (gzip), chunking for large datasets, and multiple formats
  - Asynchronous processing support for long-running operations

## [0.1.0] - 2024-09-15

### Initial Features

- Initial release of Kernel Context Server
- MCP protocol implementation for kernel analysis
- Basic call graph analysis using tree-sitter
- PostgreSQL database with pgvector for semantic search
- Docker Compose deployment configuration
- Basic API endpoints for code search and symbol analysis
