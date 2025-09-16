# KCS Documentation

Welcome to the Kernel Context Server (KCS) documentation. This directory contains
comprehensive guides for installing, configuring, and using KCS.

## Quick Links

- **🚀 [Installation Guide](INSTALLATION.md)** - Complete installation instructions for all platforms
- **📖 [Main README](../README.md)** - Project overview, quick start, and API reference
- **🔧 [Troubleshooting](../README.md#troubleshooting)** - Common issues and solutions

## Getting Started

### New Users

1. **Install KCS**: Follow the [Installation Guide](INSTALLATION.md) for your platform
2. **Quick Setup**: Run `bash tools/quick-setup.sh` for automated setup
3. **First Steps**: Try the examples in the main [README](../README.md#quick-start)

### Developers

1. **Development Setup**: Use local installation method in [INSTALLATION.md](INSTALLATION.md)
2. **Project Structure**: See [README](../README.md#development) for codebase overview
3. **Contributing**: Check [CONTRIBUTING.md](../CONTRIBUTING.md) (if available)

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [README.md](../README.md) | Main project documentation with API examples |
| [INSTALLATION.md](INSTALLATION.md) | Complete installation guide for all platforms |
| [LICENSE](../LICENSE) | MIT license terms |

### Development

| Document | Description |
|----------|-------------|
| [Makefile](../Makefile) | Build and development commands |
| [pyproject.toml](../pyproject.toml) | Python package configuration |
| [Cargo.toml](../Cargo.toml) | Rust workspace configuration |

### Configuration

| File | Description |
|------|-------------|
| [.env.example](../.env.example) | Environment variables template |
| [docker-compose.yml](../docker-compose.yml) | Container orchestration |
| [Dockerfile.mcp](../Dockerfile.mcp) | MCP server container image |

## Architecture Overview

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

## Common Use Cases

### 1. Trying KCS for the First Time

```bash
# Quick setup with Docker
git clone https://github.com/your-org/kcs.git
cd kcs
bash tools/quick-setup.sh

# Test the API
curl http://localhost:8080/health
```

### 2. Local Development

```bash
# Full local setup
bash tools/quick-setup.sh --local

# Start development server
source .venv/bin/activate
export PATH="$PWD/target/release:$PATH"
kcs-mcp --host 0.0.0.0 --port 8080 --log-level debug
```

### 3. Kernel Analysis

```bash
# Install kernel dependencies
sudo apt-get install flex bison build-essential libssl-dev libelf-dev

# Index a kernel
export DATABASE_URL="postgresql://kcs:postgres_i_hardly_knew_ya@localhost:5432/kcs"
export PYTHONPATH=""
tools/index_kernel.sh ~/src/linux
```

### 4. Production Deployment

```bash
# Automated production setup
sudo bash tools/setup/install.sh --type production

# Or manual systemd service setup
# See INSTALLATION.md for details
```

## Troubleshooting

If you encounter issues:

1. **Check the [Troubleshooting section](../README.md#troubleshooting)** in the main README
2. **Verify prerequisites** for your installation method
3. **Check service logs**:

   ```bash
   # Docker
   docker compose logs kcs-mcp

   # Local development
   journalctl -u kcs -f
   ```

4. **Test components individually**:

   ```bash
   # Database connection
   psql "postgresql://kcs:postgres_i_hardly_knew_ya@localhost:5432/kcs" -c "SELECT 1;"

   # MCP server health
   curl http://localhost:8080/health

   # Analysis tools
   kcs-parser --version
   ```

## Contributing

We welcome contributions! Please:

1. **Read the installation guide** to set up your development environment
2. **Run tests** before submitting: `make test`
3. **Follow code style**: `make lint && make format`
4. **Update documentation** for new features

## Support

- **GitHub Issues**: [Report bugs](https://github.com/your-org/kcs/issues)
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: This directory contains comprehensive guides

## Project Status

KCS is under active development. Key features:

- ✅ **MCP Protocol Support** - Complete implementation
- ✅ **Docker Deployment** - Production-ready containers
- ✅ **Kernel Parsing** - Tree-sitter and Clang support
- ✅ **Database Storage** - PostgreSQL with pgvector
- 🚧 **Performance Optimization** - Ongoing improvements
- 🚧 **Advanced Analytics** - Additional analysis tools

---

**Built for kernel developers, by kernel developers** 🐧

For the latest updates, check the main [README](../README.md) and [GitHub repository](https://github.com/your-org/kcs).
