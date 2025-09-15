# Kernel Context Server (KCS)

> Ground-truth Linux kernel analysis via MCP protocol for AI coding assistants

[![CI Status](https://github.com/your-org/kcs/workflows/CI/badge.svg)](https://github.com/your-org/kcs/actions)
[![Docker Image](https://img.shields.io/docker/v/kcs/mcp-server?label=docker)](https://hub.docker.com/r/kcs/mcp-server)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

KCS provides comprehensive Linux kernel analysis through the Model Context Protocol (MCP),
enabling AI assistants to understand kernel code structure, dependencies, and impact
analysis with ground-truth accuracy.

## Features

🔍 **Semantic Code Search** - Search kernel code by natural language or technical terms\
📊 **Call Graph Analysis** - Understand function relationships and dependencies\
🔌 **Entry Point Detection** - Identify syscalls, ioctls, and kernel interfaces\
💥 **Impact Analysis** - Assess blast radius of code changes\
📖 **Auto-generated Summaries** - AI-powered documentation for complex kernel code\
🔒 **Read-only & Safe** - Never modifies kernel source code\
⚡ **High Performance** - Sub-600ms query response times\
🔗 **Citation Support** - All results include file:line references

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **Python 3.11+** with `uv` package manager
- **Rust 1.75+** (for building from source)
- **PostgreSQL 15+** with pgvector extension
- **Linux kernel source** (for indexing)

### Option 1: Docker Compose (Recommended)

1. **Clone and Start Infrastructure**

   ```bash
   git clone https://github.com/randomparity/kcs.git
   cd kcs
   make docker-compose-up  # Starts PostgreSQL and Redis
   ```

2. **Start the MCP Server**

   ```bash
   # Option A: Use convenient make targets
   make docker-compose-up-app      # Start MCP server + infrastructure
   make docker-compose-up-all      # Start everything including monitoring

   # Option B: Use docker compose directly
   docker compose --profile app up -d
   docker compose --profile app --profile monitoring up -d
   ```

3. **Verify Services**

   ```bash
   # Check all services are running
   docker compose ps

   # Test MCP server health
   curl http://localhost:8080/health
   ```

4. **Index Your Kernel**

   ```bash
   # Index a Linux kernel repository
   tools/index_kernel.sh ~/src/linux

   # Or with custom configuration
   tools/index_kernel.sh -c arm64:defconfig ~/src/linux
   ```

5. **Test the API**

   ```bash
   # Search for memory management code
   curl -X POST http://localhost:8080/mcp/tools/search_code \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
     -d '{"query": "memory allocation", "topK": 5}'

   # Get symbol information
   curl -X POST http://localhost:8080/mcp/tools/get_symbol \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
     -d '{"symbol": "sys_read"}'
   ```

### Option 2: Local Development Setup

1. **Setup Development Environment**

   ```bash
   make setup  # Creates venv, installs dependencies, builds Rust components
   source .venv/bin/activate
   ```

2. **Start Database**

   ```bash
   make db-start  # Starts PostgreSQL with pgvector
   make db-migrate  # Runs database migrations
   ```

3. **Start KCS Server**

   ```bash
   make dev  # Starts development server on port 8080
   ```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Assistant  │◄──►│   MCP Protocol   │◄──►│   KCS Server    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────┐              │
                       │   Call Graph    │◄─────────────┤
                       │   (PostgreSQL)  │              │
                       └─────────────────┘              │
                                                        │
                       ┌─────────────────┐              │
                       │  Rust Parsers   │◄─────────────┤
                       │  (Tree-sitter)  │              │
                       └─────────────────┘              │
                                                        │
                       ┌─────────────────┐              │
                       │ Linux Kernel    │◄─────────────┘
                       │ Source Code     │
                       └─────────────────┘
```

## MCP API Endpoints

### Core Analysis Tools

| Tool | Description | Example |
|------|-------------|---------|
| `search_code` | Semantic/lexical code search | Find memory management functions |
| `get_symbol` | Symbol information with summary | Get details for `vfs_read` |
| `who_calls` | Find callers of a function | Who calls `kmalloc`? |
| `list_dependencies` | Find function callees | What does `sys_open` call? |
| `entrypoint_flow` | Trace from entry to implementation | Follow syscall path |
| `impact_of` | Analyze change blast radius | Impact of modifying `mm/mmap.c` |
| `search_docs` | Search kernel documentation | Find memory barrier docs |
| `owners_for` | Find code maintainers | Who maintains `fs/ext4/`? |

### Example Queries

```bash
# Search for networking code
curl -X POST http://localhost:8080/mcp/tools/search_code \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "query": "tcp socket implementation",
    "topK": 10,
    "config": "x86_64:defconfig"
  }'

# Analyze who calls a critical function
curl -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "symbol": "schedule",
    "depth": 2,
    "config": "x86_64:defconfig"
  }'

# Trace syscall implementation
curl -X POST http://localhost:8080/mcp/tools/entrypoint_flow \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "entry": "__NR_read",
    "config": "x86_64:defconfig"
  }'

# Impact analysis for a code change
curl -X POST http://localhost:8080/mcp/tools/impact_of \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "files": ["mm/mmap.c"],
    "symbols": ["mmap_region"],
    "config": "x86_64:defconfig"
  }'
```

## Indexing Process

### Full Kernel Index

Index a complete kernel repository:

```bash
# Basic indexing
tools/index_kernel.sh ~/src/linux

# With specific configuration
tools/index_kernel.sh --config arm64:defconfig ~/src/linux

# Custom output directory and parallel jobs
tools/index_kernel.sh --output /data/kcs-index --jobs 8 ~/src/linux

# Incremental update (faster for git changes)
tools/index_kernel.sh --incremental ~/src/linux
```

### Indexing Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Kernel configuration | `x86_64:defconfig`, `arm64:allnoconfig` |
| `--output` | Output directory | `/data/kcs-index` |
| `--jobs` | Parallel workers | `8` (default: 4) |
| `--incremental` | Faster updates | For git changes |
| `--no-clang` | Skip semantic analysis | Faster but less accurate |
| `--verbose` | Detailed logging | For debugging |
| `--dry-run` | Show what would be done | Test configuration |

### Performance Targets

- **Full Index**: ≤ 20 minutes (constitutional requirement)
- **Incremental Update**: ≤ 3 minutes
- **Query Response**: p95 ≤ 600ms
- **Database Size**: < 20GB for 6 configurations

## Development

### Project Structure

```
kcs/
├── src/
│   ├── rust/                # Performance-critical Rust components
│   │   ├── kcs-parser/      # Tree-sitter + clang parsing
│   │   ├── kcs-extractor/   # Entry point detection
│   │   ├── kcs-graph/       # Call graph algorithms
│   │   ├── kcs-impact/      # Impact analysis
│   │   └── kcs-drift/       # Spec vs code drift detection
│   ├── python/              # MCP server and integration
│   │   ├── kcs_mcp/         # FastAPI MCP protocol server
│   │   └── kcs_summarizer/  # LLM-powered summaries
│   └── sql/                 # Database schema and migrations
├── tests/                   # Comprehensive test suite
├── tools/                   # Development and deployment tools
└── docs/                    # Documentation
```

### Development Commands

```bash
# Setup development environment
make setup

# Code quality checks
make check          # Run all quality checks
make lint           # Linting (ruff, mypy, clippy)
make format         # Code formatting
make test           # Full test suite

# Testing
make test-unit           # Unit tests
make test-integration    # Integration tests
make test-contract       # API contract tests
make test-performance    # Performance benchmarks
make test-system         # Full system test

# Build and run
make build          # Build all components
make dev           # Start development server
make benchmark     # Performance benchmarks

# Database operations
make db-start      # Start PostgreSQL
make db-migrate    # Run migrations
make db-stop       # Stop database

# CI/CD simulation
make ci            # Run CI pipeline locally
make pre-commit    # Pre-commit hooks
```

### Testing with Real Kernels

KCS includes comprehensive test fixtures:

```bash
# Test with mini kernel fixture (fast)
make test-mini-kernel

# Run system test with your kernel
python tools/run_system_test.py

# Performance testing
make benchmark-k6    # Load testing with k6
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://kcs:kcs_password@localhost:5432/kcs` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `JWT_SECRET` | Authentication secret | `dev_jwt_secret_change_in_production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `KCS_KERNEL_PATH` | Default kernel path | - |

### Kernel Configurations

KCS supports multiple kernel configurations simultaneously:

```bash
# Index multiple configurations
tools/index_kernel.sh -c x86_64:defconfig ~/src/linux
tools/index_kernel.sh -c arm64:defconfig ~/src/linux
tools/index_kernel.sh -c x86_64:allnoconfig ~/src/linux

# Query specific configuration
curl -X POST .../search_code -d '{"query": "...", "config": "arm64:defconfig"}'
```

## Monitoring

### Health Checks

```bash
# Service health
curl http://localhost:8080/health

# Detailed metrics (Prometheus format)
curl http://localhost:8080/metrics

# MCP protocol info
curl http://localhost:8080/mcp/resources
```

### Grafana Dashboards

Start monitoring stack:

```bash
docker compose --profile monitoring up -d

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

## Constitutional Requirements

KCS adheres to strict constitutional requirements:

✅ **Read-Only**: Never modifies kernel source code
✅ **Citations Required**: All results include file:line references
✅ **MCP-First**: All features accessible via Model Context Protocol
✅ **Config-Aware**: Results tagged with kernel configuration
✅ **Performance**: Index ≤20min, queries p95 ≤600ms

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check database is running
docker compose ps postgres

# Check connection
psql $DATABASE_URL -c "SELECT 1;"

# Reset database
make db-stop && make db-start && make db-migrate
```

#### Parsing Failures

```bash
# Check Rust tools are built
make build-rust

# Test with single file
./target/debug/kcs-parser file ~/src/linux/kernel/fork.c

# Enable verbose logging
tools/index_kernel.sh --verbose ~/src/linux
```

#### Performance Issues

```bash
# Check database indexes
psql $DATABASE_URL -c "SELECT * FROM pg_stat_user_indexes WHERE idx_scan < 100;"

# Monitor query performance
curl http://localhost:8080/metrics | grep query_duration

# Run performance optimization
python tools/performance_optimization.py --analyze
```

### Getting Help

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/kcs/issues)
- **Documentation**: [Complete API docs](docs/)
- **Contributing**: [Contribution guidelines](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built for kernel developers, by kernel developers** 🐧

For questions or support, please open an issue or check our documentation.
