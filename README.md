# Kernel Context Server (KCS)

> Ground-truth Linux kernel analysis via MCP protocol for AI coding assistants

[![CI Status](https://github.com/randomparity/kcs/workflows/CI/badge.svg)](https://github.com/randomparity/kcs/actions)
[![Docker Image](https://img.shields.io/docker/v/kcs/mcp-server?label=docker)](https://hub.docker.com/r/kcs/mcp-server)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

KCS provides comprehensive Linux kernel analysis through the Model Context Protocol (MCP),
enabling AI assistants to understand kernel code structure, dependencies, and impact
analysis with ground-truth accuracy.

## Features

🔍 **Semantic Code Search** - Search kernel code by natural language or technical terms\
📊 **Call Graph Analysis** - Real-time function relationships and dependencies using tree-sitter parsing\
🔌 **Entry Point Detection** - Identify syscalls, ioctls, and kernel interfaces\
💥 **Impact Analysis** - Assess blast radius of code changes\
📖 **Auto-generated Summaries** - AI-powered documentation for complex kernel code\
🔒 **Read-only & Safe** - Never modifies kernel source code\
⚡ **High Performance** - Sub-600ms query response times\
🔗 **Citation Support** - All results include file:line references

## Quick Start

> **🚀 New to KCS?** Run our quick setup script: `bash tools/quick-setup.sh`
>
> **📖 Need detailed instructions?** See the complete [Installation Guide](docs/INSTALLATION.md)

### System Prerequisites

**Required for all installation methods:**

- **Linux or macOS** (Windows via WSL2)
- **Git** for source code management
- **4GB+ RAM** (8GB+ recommended for large kernels)
- **10GB+ disk space** for data and indexes

**For Docker installation (recommended):**

- **Docker 20.10+** and **Docker Compose v2**
- No other dependencies needed

**For local development:**

- **Python 3.11+** with `pip` and `venv`
- **Rust 1.75+** with `cargo`
- **PostgreSQL 15+** with pgvector extension
- **Node.js 18+** (for some development tools)

**For kernel indexing:**

- **Linux kernel source code** (git clone of linux.git)
- **Kernel build dependencies** (flex, bison, build-essential)
- **Clang 15+** (optional, for enhanced semantic analysis)

### Option 1: Docker Compose (Recommended)

1. **Install System Dependencies**

   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-v2 git curl flex bison build-essential libssl-dev libelf-dev bear

   # macOS (with Homebrew)
   brew install docker docker-compose git

   # Start Docker service (Linux only)
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER  # Re-login after this
   ```

2. **Clone and Setup Environment**

   ```bash
   git clone https://github.com/randomparity/kcs.git
   cd kcs

   # Copy environment configuration (includes database passwords, ports, etc.)
   cp .env.example .env

   # Optional: Edit .env to customize settings
   # Key settings: ports, data directories, authentication tokens
   ```

3. **Setup Data Persistence**

   ```bash
   # Create host directories for persistent data storage
   make create-data-dirs

   # This creates: ./appdata/{postgres,redis,grafana,prometheus,kcs}
   # Data persists across container restarts
   ```

4. **Start Services**

   ```bash
   # Start all application services (PostgreSQL, Redis, KCS MCP server)
   make docker-compose-up-app

   # Or with monitoring (adds Grafana, Prometheus)
   make docker-compose-up-all

   # Alternative: Use docker compose directly
   docker compose --profile app up -d
   ```

5. **Verify Installation**

   ```bash
   # Check all services are healthy
   docker compose ps

   # Test MCP server (should return {"status":"healthy"})
   curl http://localhost:8080/health

   # Check logs if any issues
   docker compose logs kcs-mcp
   ```

6. **Install Kernel Analysis Tools** ⚠️

   The Docker setup only runs the MCP server. For kernel indexing, you need additional tools:

   ```bash
   # Build KCS analysis tools locally
   cargo build --release

   # Install to default rust directory (~/.cargo/bin)
   cp target/release/kcs-* ~/.cargo/bin/

   # (or) Install to user local binary directory
   cp target/release/kcs-* ~/.local/bin/

   # Verify tools are available
   which kcs-parser kcs-extractor kcs-graph kcs-impact
   ```

7. **Index Your First Kernel**

   ```bash
   # Clone a Linux kernel (if you don't have one)
   git clone --depth 1 https://github.com/torvalds/linux.git ~/src/linux

   # Set environment for indexing tools
   export DATABASE_URL="postgresql://kcs:postgres_i_hardly_knew_ya@localhost:5432/kcs"
   export PYTHONPATH=""

   # Index the kernel (takes 10-30 minutes depending on system)
   tools/index_kernel.sh ~/src/linux

   # For faster indexing without semantic analysis:
   tools/index_kernel.sh --no-clang ~/src/linux
   ```

8. **Test the API**

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

Complete local installation without Docker.

1. **Install System Dependencies**

   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y \
     git curl build-essential pkg-config libssl-dev \
     python3 python3-pip python3-venv \
     postgresql postgresql-contrib postgresql-15-pgvector \
     flex bison libelf-dev \
     clang

   # macOS (with Homebrew)
   brew install git curl pkg-config openssl python postgresql pgvector \
     flex bison clang

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   rustup component add clippy rustfmt
   ```

2. **Setup PostgreSQL Database**

   ```bash
   # Start PostgreSQL service
   sudo systemctl enable --now postgresql  # Linux
   # brew services start postgresql         # macOS

   # Create KCS database and user
   sudo -u postgres psql -c "CREATE DATABASE kcs;"
   sudo -u postgres psql -c "CREATE USER kcs WITH PASSWORD 'kcs_password';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kcs TO kcs;"
   sudo -u postgres psql -c "ALTER USER kcs CREATEDB;"

   # Install pgvector extension
   sudo -u postgres psql -d kcs -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. **Clone and Build KCS**

   ```bash
   git clone https://github.com/randomparity/kcs.git
   cd kcs

   # Setup Python environment
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel

   # Install Python dependencies and KCS
   pip install -e .

   # Build Rust components
   cargo build --release

   # Install tools to local bin (optional - adds to ~/.local/bin)
   mkdir -p ~/.local/bin
   cp target/release/kcs-* ~/.local/bin/
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Setup Configuration**

   ```bash
   # Copy environment configuration
   cp .env.example .env

   # Edit .env to match your setup - key changes:
   # POSTGRES_PASSWORD=kcs_password
   # DATABASE_URL=postgresql://kcs:kcs_password@localhost:5432/kcs

   # Create data directories
   make create-data-dirs
   ```

5. **Run Database Migrations**

   ```bash
   # Load environment variables
   export $(grep -v '^#' .env | xargs)

   # Run migrations (if migration script exists)
   if [ -f "tools/setup/migrate.sh" ]; then
     bash tools/setup/migrate.sh
   fi
   ```

6. **Start KCS Server**

   ```bash
   # Activate environment
   source .venv/bin/activate
   export $(grep -v '^#' .env | xargs)

   # Start development server
   kcs-mcp --host 0.0.0.0 --port 8080 --log-level info

   # Or use Python module directly
   python -m kcs_mcp.cli --host 0.0.0.0 --port 8080
   ```

7. **Verify Installation**

   ```bash
   # Test server health
   curl http://localhost:8080/health

   # Test analysis tools
   kcs-parser --version
   kcs-extractor --help
   kcs-graph --help
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
# Basic indexing with call graph extraction
tools/index_kernel.sh ~/src/linux

# With specific configuration
tools/index_kernel.sh --config arm64:defconfig ~/src/linux

# Custom output directory and parallel jobs
tools/index_kernel.sh --output /data/kcs-index --jobs 8 ~/src/linux

# Incremental update (faster for git changes)
tools/index_kernel.sh --incremental ~/src/linux

# Use kcs-parser directly with call graph extraction
kcs-parser --include-calls directory ~/src/linux/fs/
```

### Indexing Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Kernel configuration | `x86_64:defconfig`, `arm64:allnoconfig` |
| `--output` | Output directory | `/data/kcs-index` |
| `--jobs` | Parallel workers | `8` (default: 4) |
| `--incremental` | Faster updates | For git changes |
| `--no-clang` | Skip semantic analysis | Faster but less accurate |
| `--include-calls` | Enable call graph extraction | Enhanced analysis with function relationships |
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
cargo bench --bench call_graph_bench  # Call graph extraction benchmarks
```

## Configuration

### Environment Setup

KCS uses environment variables for configuration. Copy the example file and customize as needed:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Key Environment Variables

#### Database Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_DB` | Database name | `kcs` |
| `POSTGRES_USER` | Database user | `kcs` |
| `POSTGRES_PASSWORD` | Database password | `kcs_dev_password` |
| `DATABASE_URL` | Full connection string | Auto-generated |

#### Server Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Authentication secret | `dev_jwt_secret...` |
| `KCS_PORT` | Server port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `KCS_WORKERS` | Worker processes | `4` |

#### Port Mapping

| Variable | Description | Default |
|----------|-------------|---------|
| `KCS_EXTERNAL_PORT` | KCS server port (host) | `8080` |
| `POSTGRES_EXTERNAL_PORT` | PostgreSQL port (host) | `5432` |
| `REDIS_EXTERNAL_PORT` | Redis port (host) | `6379` |
| `GRAFANA_EXTERNAL_PORT` | Grafana port (host) | `3000` |

#### Performance Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `KCS_MEMORY_LIMIT` | Docker memory limit | `1g` |
| `POSTGRES_MEMORY_LIMIT` | PostgreSQL memory | `2g` |
| `DB_POOL_SIZE` | Connection pool size | `20` |

### Production Configuration

For production, ensure you change these critical settings in your `.env` file:

```bash
# Security
POSTGRES_PASSWORD=your_secure_password_here
JWT_SECRET=your_64_character_random_string_here
GRAFANA_ADMIN_PASSWORD=your_grafana_password_here

# Environment
ENVIRONMENT=production
DEBUG=false
SECURE_COOKIES=true
CORS_ORIGINS=https://your-domain.com

# Performance
KCS_WORKERS=8
DB_POOL_SIZE=50
POSTGRES_MEMORY_LIMIT=4g
KCS_MEMORY_LIMIT=2g
```

### Data Persistence

KCS uses host bind mounts for data persistence, ensuring your data survives container restarts and updates.

#### Data Directory Structure

```
./data/                          # Base data directory
├── postgres/                    # PostgreSQL database files
├── redis/                       # Redis persistence files
├── grafana/                     # Grafana dashboards and settings
├── prometheus/                  # Prometheus metrics data
├── kcs-index/                   # Kernel index and analysis data
├── cache/                       # Application cache
└── logs/                        # Service logs
    ├── postgres/
    └── redis/
```

#### Data Management Commands

```bash
# Create data directories
make create-data-dirs

# Backup all data
make backup-data

# Clean all data (WARNING: destructive!)
make clean-data

# Restore from backup
make restore-data BACKUP_PATH=backups/backup-20241215-143022
```

#### Customizing Data Locations

Edit your `.env` file to change data locations:

```bash
# Custom data directory
KCS_DATA_DIR=/var/lib/kcs

# Individual service directories
POSTGRES_DATA_DIR=/data/postgresql
REDIS_DATA_DIR=/data/redis
KCS_INDEX_DATA_DIR=/data/kcs-indexes
```

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

### Common Installation Issues

#### Missing System Dependencies

**Problem**: `flex: not found` or `bison: not found` during kernel indexing

```bash
# Solution: Install kernel build dependencies
sudo apt-get install -y flex bison build-essential libssl-dev libelf-dev

# Verify installation
which flex bison gcc
```

**Problem**: `kcs-parser: command not found`

```bash
# Solution 1: Build and add to PATH
cargo build --release
export PATH="$PWD/target/release:$PATH"

# Solution 2: Install to local bin
mkdir -p ~/.local/bin
cp target/release/kcs-* ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
which kcs-parser kcs-extractor kcs-graph
```

**Problem**: `psql: command not found`

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-client

# macOS
brew install postgresql

# Verify installation
psql --version
```

#### Docker Issues

**Problem**: Docker containers fail to start

```bash
# Check Docker is running
sudo systemctl status docker

# Check container status
docker compose ps

# View specific container logs
docker compose logs postgres
docker compose logs kcs-mcp

# Restart services
docker compose down && docker compose --profile app up -d
```

**Problem**: Port conflicts (port already in use)

```bash
# Check what's using the port
sudo lsof -i :8080
sudo lsof -i :5432

# Edit .env to use different ports
# KCS_EXTERNAL_PORT=8081
# POSTGRES_EXTERNAL_PORT=5433

# Restart with new ports
docker compose down && docker compose --profile app up -d
```

#### Local Database Connection Issues

**Problem**: Cannot connect to database

```bash
# Check PostgreSQL is running
docker compose ps postgres
# or for local install:
sudo systemctl status postgresql

# Test connection manually
psql "postgresql://kcs:postgres_i_hardly_knew_ya@localhost:5432/kcs" -c "SELECT 1;"

# Check .env database URL matches docker-compose.yml
grep DATABASE_URL .env
grep POSTGRES_PASSWORD .env
```

**Problem**: Database authentication failed

```bash
# Verify password in .env matches docker-compose.yml
grep POSTGRES_PASSWORD .env
grep POSTGRES_PASSWORD docker-compose.yml

# Reset database (WARNING: destroys data)
docker compose down -v
make create-data-dirs
docker compose --profile app up -d
```

### Runtime Issues

#### Kernel Indexing Problems

**Problem**: `PYTHONPATH: unbound variable`

```bash
# Solution: Set environment variables before indexing
export PYTHONPATH=""
export DATABASE_URL="postgresql://kcs:postgres_i_hardly_knew_ya@localhost:5432/kcs"
export PATH="$PWD/target/release:$PATH"

# Then run indexing
tools/index_kernel.sh ~/src/linux
```

**Problem**: Kernel indexing fails with compilation errors

```bash
# Option 1: Use tree-sitter only (faster, less accurate)
tools/index_kernel.sh --no-clang ~/src/linux

# Option 2: Install full kernel build dependencies
sudo apt-get install -y \
  flex bison build-essential libssl-dev libelf-dev \
  bc kmod cpio initramfs-tools

# Option 3: Use pre-built compile_commands.json (if available)
cp /path/to/existing/compile_commands.json ~/src/linux/
tools/index_kernel.sh ~/src/linux
```

**Problem**: Out of memory during indexing

```bash
# Reduce parallel jobs
tools/index_kernel.sh --jobs 2 ~/src/linux

# Use incremental indexing for updates
tools/index_kernel.sh --incremental ~/src/linux

# Monitor memory usage
htop  # or top
```

#### API/Server Issues

**Problem**: MCP server returns 401 Unauthorized

```bash
# Check authentication token
curl -H "Authorization: Bearer dev_jwt_secret_change_in_production" \
  http://localhost:8080/health

# Verify JWT_SECRET in .env
grep JWT_SECRET .env

# Use development token for testing
AUTH_TOKEN="dev-token"  # This always works in dev mode
```

**Problem**: Slow query responses

```bash
# Check database indexes
docker compose exec postgres psql -U kcs -d kcs -c "
SELECT schemaname,tablename,attname,n_distinct,correlation
FROM pg_stats WHERE tablename IN ('symbols', 'call_edges', 'files');"

# Check query performance
curl http://localhost:8080/metrics | grep query_duration

# Optimize database
docker compose exec postgres psql -U kcs -d kcs -c "
ANALYZE;
REINDEX DATABASE kcs;"
```

### Development Issues

#### Build Problems

**Problem**: Rust compilation fails

```bash
# Update Rust to latest stable
rustup update stable
rustup default stable

# Clean and rebuild
cargo clean
cargo build --release

# Check for specific errors
cargo check --all-targets
```

**Problem**: Python package installation fails

```bash
# Ensure using correct Python version
python3 --version  # Should be 3.11+

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Install with verbose output
pip install -e . -v

# Clear pip cache if needed
pip cache purge
```

#### Test Failures

**Problem**: Tests fail with database errors

```bash
# Ensure test database is clean
make test-db-reset

# Run tests with verbose output
python -m pytest tests/ -v -s

# Run specific failing test
python -m pytest tests/contract/test_search_code.py -v
```

### Debugging Commands

#### Useful Debug Commands

```bash
# Check all service health
make health-check

# Verify environment configuration
export $(grep -v '^#' .env | xargs)
echo "Database: $DATABASE_URL"
echo "JWT Secret: ${JWT_SECRET:0:10}..."  # Show first 10 chars only

# Test with different .env files
cp .env.example .env.test
# Edit .env.test with test settings
docker compose --env-file .env.test config
```

#### Debug Parsing Failures

```bash
# Check Rust tools are built
make build-rust

# Test with single file
./target/debug/kcs-parser file ~/src/linux/kernel/fork.c

# Enable verbose logging
tools/index_kernel.sh --verbose ~/src/linux
```

#### Debug Performance Issues

```bash
# Check database indexes
psql $DATABASE_URL -c "SELECT * FROM pg_stat_user_indexes WHERE idx_scan < 100;"

# Monitor query performance
curl http://localhost:8080/metrics | grep query_duration

# Run performance optimization
python tools/performance_optimization.py --analyze
```

#### Docker Database Connection Issues

```bash
# Check database is running
docker compose ps postgres

# Check connection with your environment variables
source .env
psql $DATABASE_URL -c "SELECT 1;"

# Reset database
make db-stop && make db-start && make db-migrate

# Check environment variables are loaded
docker compose config | grep -A 5 -B 5 DATABASE_URL

# Validate environment file
make validate-env  # (if available)
```

#### Data Persistence Issues

```bash
# Check data directory permissions
ls -la data/

# Fix ownership issues (run as your user)
sudo chown -R $USER:$USER data/

# Check disk space
df -h data/

# Verify bind mounts are working
docker compose exec postgres ls -la /var/lib/postgresql/data/
docker compose exec redis ls -la /data/

# Reset data if corrupted
make clean-data  # WARNING: destructive!
make create-data-dirs
```

#### Environment Variable Issues

```bash
# Test environment variable loading
docker compose config

# Verify specific variables
echo "Database: $DATABASE_URL"
echo "JWT Secret: ${JWT_SECRET:0:10}..."  # Show first 10 chars only

# Test with different .env files
cp .env.example .env.test
# Edit .env.test with test settings
docker compose --env-file .env.test config
```

#### Kernel Parsing Failures

```bash
# Check Rust tools are built
make build-rust

# Test with single file
./target/debug/kcs-parser file ~/src/linux/kernel/fork.c

# Enable verbose logging
tools/index_kernel.sh --verbose ~/src/linux
```

#### Query Performance Issues

```bash
# Check database indexes
psql $DATABASE_URL -c "SELECT * FROM pg_stat_user_indexes WHERE idx_scan < 100;"

# Monitor query performance
curl http://localhost:8080/metrics | grep query_duration

# Run performance optimization
python tools/performance_optimization.py --analyze
```

### Getting Help

- **Installation Issues**: See [Installation Guide](docs/INSTALLATION.md) for platform-specific instructions
- **Runtime Problems**: Check the comprehensive [Troubleshooting](#troubleshooting) section above
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/kcs/issues)
- **Documentation**: [Complete API docs](docs/)
- **Contributing**: [Contribution guidelines](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built for kernel developers, by kernel developers** 🐧

For questions or support, please open an issue or check our documentation.
