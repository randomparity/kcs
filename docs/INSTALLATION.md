# KCS Installation Guide

Complete installation instructions for Kernel Context Server (KCS)
across different platforms and use cases.

## Overview

KCS can be installed in several ways depending on your needs:

1. **Docker Compose** (Recommended) - Easiest for trying KCS
2. **Local Development** - Full control for development work
3. **Production Deployment** - Systemd service with security hardening

## Prerequisites by Installation Type

### All Installations

- **Operating System**: Linux (Ubuntu/Debian/RHEL/Arch) or macOS
- **Hardware**: 4GB+ RAM, 10GB+ disk space
- **Git**: For cloning repositories

### Docker Installation

```bash
# Ubuntu/Debian
sudo apt-get install docker.io docker-compose-v2

# RHEL/CentOS/Fedora
sudo dnf install docker docker-compose

# macOS
brew install docker docker-compose

# Start Docker (Linux only)
sudo systemctl enable --now docker
sudo usermod -aG docker $USER  # Logout/login after this
```

### Local Development

```bash
# Ubuntu/Debian - Full development stack
sudo apt-get update && sudo apt-get install -y \
  git curl wget build-essential pkg-config \
  python3 python3-pip python3-venv \
  postgresql postgresql-contrib postgresql-15-pgvector \
  flex bison libssl-dev libelf-dev \
  clang llvm

# macOS - Using Homebrew
brew install git curl wget pkg-config openssl \
  python@3.11 postgresql pgvector \
  flex bison clang

# Install Rust (all platforms)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup component add clippy rustfmt
```

### Kernel Build Dependencies (for indexing)

```bash
# Ubuntu/Debian
sudo apt-get install -y \
  flex bison build-essential libssl-dev libelf-dev \
  bc kmod cpio initramfs-tools

# RHEL/CentOS/Fedora
sudo dnf install -y \
  flex bison gcc make openssl-devel elfutils-libelf-devel \
  bc kmod

# macOS
brew install flex bison
```

## Installation Methods

### Method 1: Docker Compose (Recommended for Testing)

**Pros**: Easy setup, isolated environment, includes all services
**Cons**: Requires Docker, kernel indexing tools need separate build

1. **Install Docker** (see prerequisites above)

2. **Clone and Setup**

   ```bash
   git clone https://github.com/your-org/kcs.git
   cd kcs

   # Copy environment configuration
   cp .env.example .env

   # Create persistent data directories
   make create-data-dirs
   ```

3. **Start Services**

   ```bash
   # Start application services (PostgreSQL, Redis, KCS MCP server)
   make docker-compose-up-app

   # Or with monitoring (adds Grafana, Prometheus)
   make docker-compose-up-all
   ```

4. **Verify Installation**

   ```bash
   # Check services are running
   docker compose ps

   # Test MCP server
   curl http://localhost:8080/health
   # Should return: {"status":"healthy","version":"1.0.0","indexed_at":null}
   ```

5. **Build Indexing Tools** (for kernel analysis)

   ```bash
   # Install Rust if not already installed
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

   # Install kernel build dependencies
   sudo apt-get install -y flex bison build-essential libssl-dev libelf-dev

   # Build KCS analysis tools
   cargo build --release

   # Add to PATH permanently
   echo 'export PATH="'$PWD'/target/release:$PATH"' >> ~/.bashrc
   source ~/.bashrc

   # Verify tools
   which kcs-parser kcs-extractor kcs-graph
   ```

6. **Index Your First Kernel** (optional)

   ```bash
   # Clone Linux kernel if needed
   git clone --depth 1 https://github.com/torvalds/linux.git ~/src/linux

   # Set environment variables
   export DATABASE_URL="postgresql://kcs:postgres_i_hardly_knew_ya@localhost:5432/kcs"
   export PYTHONPATH=""

   # Index kernel (20-30 minutes)
   tools/index_kernel.sh ~/src/linux

   # Or faster tree-sitter only mode
   tools/index_kernel.sh --no-clang ~/src/linux
   ```

### Method 2: Local Development (Full Control)

**Pros**: Full control, easier debugging, native performance
**Cons**: More setup steps, manual dependency management

1. **Install System Dependencies** (see prerequisites above)

2. **Setup PostgreSQL Database**

   ```bash
   # Start PostgreSQL
   sudo systemctl enable --now postgresql  # Linux
   # brew services start postgresql         # macOS

   # Create database and user
   sudo -u postgres createdb kcs
   sudo -u postgres createuser kcs
   sudo -u postgres psql -c "ALTER USER kcs WITH PASSWORD 'kcs_password';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kcs TO kcs;"
   sudo -u postgres psql -c "ALTER USER kcs CREATEDB;"

   # Install pgvector extension
   sudo -u postgres psql -d kcs -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. **Clone and Build KCS**

   ```bash
   git clone https://github.com/your-org/kcs.git
   cd kcs

   # Setup Python environment
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel

   # Install KCS and dependencies
   pip install -e .

   # Build Rust components
   cargo build --release

   # Install tools globally (optional)
   sudo cp target/release/kcs-* /usr/local/bin/
   ```

4. **Configure Environment**

   ```bash
   # Copy and edit configuration
   cp .env.example .env

   # Edit .env file to match your setup:
   # POSTGRES_PASSWORD=kcs_password
   # DATABASE_URL=postgresql://kcs:kcs_password@localhost:5432/kcs

   # Create data directories
   mkdir -p ./appdata/{kcs,postgres,redis}/{data,logs}
   ```

5. **Run Database Migrations**

   ```bash
   # Load environment
   export $(grep -v '^#' .env | xargs)

   # Run migrations if script exists
   [ -f "tools/setup/migrate.sh" ] && bash tools/setup/migrate.sh
   ```

6. **Start KCS Server**

   ```bash
   # Option 1: Using CLI command
   kcs-mcp --host 0.0.0.0 --port 8080 --log-level info

   # Option 2: Using Python module
   python -m kcs_mcp.cli --host 0.0.0.0 --port 8080

   # Option 3: Development server with auto-reload
   uvicorn kcs_mcp.app:app --host 0.0.0.0 --port 8080 --reload
   ```

7. **Verify Installation**

   ```bash
   # Test server
   curl http://localhost:8080/health

   # Test analysis tools
   kcs-parser --version
   kcs-extractor --help
   ```

### Method 3: Production Installation

For production deployments with systemd service management.

1. **Run Automated Installer**

   ```bash
   # Download KCS
   git clone https://github.com/your-org/kcs.git
   cd kcs

   # Run production installer
   sudo bash tools/setup/install.sh --type production
   ```

2. **Manual Production Setup** (alternative)

   ```bash
   # Create KCS user
   sudo useradd -r -s /bin/false -d /opt/kcs kcs
   sudo mkdir -p /opt/kcs

   # Install to /opt/kcs following local development steps
   # Then setup systemd service:

   sudo tee /etc/systemd/system/kcs.service << EOF
   [Unit]
   Description=Kernel Context Server
   After=network.target postgresql.service
   Requires=postgresql.service

   [Service]
   Type=simple
   User=kcs
   Group=kcs
   WorkingDirectory=/opt/kcs
   Environment=PATH=/opt/kcs/.venv/bin
   EnvironmentFile=/opt/kcs/.env
   ExecStart=/opt/kcs/.venv/bin/kcs-mcp --host 0.0.0.0 --port 8080
   Restart=always
   RestartSec=3

   [Install]
   WantedBy=multi-user.target
   EOF

   # Enable and start service
   sudo systemctl daemon-reload
   sudo systemctl enable kcs
   sudo systemctl start kcs
   ```

## Platform-Specific Notes

### Ubuntu/Debian

```bash
# Install all dependencies at once
sudo apt-get update && sudo apt-get install -y \
  git curl wget build-essential pkg-config libssl-dev \
  python3 python3-pip python3-venv \
  postgresql postgresql-contrib postgresql-15-pgvector \
  flex bison libelf-dev clang \
  docker.io docker-compose-v2

# Enable services
sudo systemctl enable --now postgresql docker
sudo usermod -aG docker $USER
```

### RHEL/CentOS/Fedora

```bash
# Install dependencies
sudo dnf update && sudo dnf install -y \
  git curl wget gcc gcc-c++ make pkg-config openssl-devel \
  python3 python3-pip \
  postgresql postgresql-server postgresql-contrib \
  flex bison elfutils-libelf-devel clang \
  docker docker-compose

# Initialize PostgreSQL (RHEL/CentOS)
sudo postgresql-setup --initdb

# Enable services
sudo systemctl enable --now postgresql docker
sudo usermod -aG docker $USER
```

### macOS

```bash
# Install dependencies with Homebrew
brew install git curl wget pkg-config openssl \
  python@3.11 postgresql pgvector \
  flex bison clang \
  docker docker-compose

# Start services
brew services start postgresql docker
```

### Arch Linux

```bash
# Install dependencies
sudo pacman -Syu && sudo pacman -S \
  git curl wget base-devel pkg-config openssl \
  python python-pip \
  postgresql postgresql-libs \
  flex bison clang \
  docker docker-compose

# Initialize PostgreSQL
sudo -u postgres initdb -D /var/lib/postgres/data

# Enable services
sudo systemctl enable --now postgresql docker
sudo usermod -aG docker $USER
```

## Post-Installation Verification

### Basic Health Checks

```bash
# 1. Check all components are installed
python3 --version      # Should be 3.11+
rustc --version        # Should be 1.75+
psql --version         # Should be 15+
docker --version       # Should be 20+

# 2. Check KCS tools are available
which kcs-parser kcs-extractor kcs-graph

# 3. Test database connection
psql "postgresql://kcs:kcs_password@localhost:5432/kcs" -c "SELECT version();"

# 4. Test MCP server
curl -s http://localhost:8080/health | jq .

# 5. Check service status (Docker)
docker compose ps

# 6. Check service status (systemd)
sudo systemctl status kcs
```

### API Testing

```bash
# Set authentication token
export AUTH_TOKEN="dev-token"  # Works in development mode

# Test search endpoint
curl -X POST http://localhost:8080/mcp/tools/search_code \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{"query": "hello world", "topK": 3}' | jq .

# Test symbol endpoint
curl -X POST http://localhost:8080/mcp/tools/get_symbol \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{"symbol": "main"}' | jq .
```

## Next Steps

1. **Index Your First Kernel**: Follow the indexing guide in the main README
2. **Explore the API**: Try the MCP endpoints with real queries
3. **Setup Monitoring**: Enable Grafana/Prometheus with `--profile monitoring`
4. **Read Documentation**: Check `docs/` for advanced configuration

## Getting Help

- **Issues**: Check existing [issues](https://github.com/your-org/kcs/issues) or create new ones
- **Troubleshooting**: See the comprehensive troubleshooting section in README.md
- **Documentation**: Browse `docs/` directory for detailed guides
- **Community**: Join discussions in GitHub Discussions

## Security Considerations

### Development Mode

- Uses default JWT secrets (insecure)
- Allows `dev-token` authentication
- Binds to all interfaces (0.0.0.0)

### Production Mode

- Generate secure JWT secrets: `openssl rand -base64 32`
- Use proper authentication tokens
- Configure firewall rules for port access
- Run as dedicated user with minimal privileges
- Enable PostgreSQL authentication and encryption

### Docker Security

- Containers run as non-root users
- Data stored in host bind mounts with proper permissions
- Network isolation between services
- Regular image updates recommended
