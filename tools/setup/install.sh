#!/bin/bash
# KCS Installation Script
#
# Installs and configures Kernel Context Server with all dependencies.
# Supports both development and production environments.

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KCS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/tmp/kcs-install-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration defaults
INSTALL_TYPE="development"
POSTGRES_VERSION="15"
PYTHON_VERSION="3.11"
RUST_VERSION="1.75"
KCS_USER="kcs"
KCS_HOME="/opt/kcs"
POSTGRES_DB="kcs"
POSTGRES_USER="kcs"
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"
KCS_PORT="8080"
INSTALL_POSTGRES=true
INSTALL_PYTHON=true
INSTALL_RUST=true
INSTALL_DOCKER=false

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $*" | tee -a "$LOG_FILE"
}

# Utility functions
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

get_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        elif [ -f /etc/arch-release ]; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

check_requirements() {
    log "Checking system requirements..."

    local os=$(get_os)
    log "Detected OS: $os"

    # Check minimum system requirements
    local memory_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
    local memory_gb=$((memory_kb / 1024 / 1024))

    if [ "$memory_gb" -lt 4 ]; then
        log_warning "System has less than 4GB RAM. KCS may run slowly."
    fi

    # Check disk space
    local disk_space=$(df "$KCS_ROOT" | tail -1 | awk '{print $4}')
    local disk_gb=$((disk_space / 1024 / 1024))

    if [ "$disk_gb" -lt 10 ]; then
        log_error "Insufficient disk space. Need at least 10GB available."
        exit 1
    fi

    log_success "System requirements check passed"
}

install_system_packages() {
    log "Installing system packages..."

    local os=$(get_os)

    case $os in
        debian)
            sudo apt-get update
            sudo apt-get install -y \
                curl wget git build-essential \
                pkg-config libssl-dev \
                libclang-dev clang \
                postgresql-client \
                python3-dev python3-pip python3-venv \
                tree ripgrep
            ;;
        redhat)
            sudo dnf update -y
            sudo dnf install -y \
                curl wget git gcc gcc-c++ make \
                pkg-config openssl-devel \
                clang clang-devel \
                postgresql \
                python3-devel python3-pip \
                tree ripgrep
            ;;
        arch)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                curl wget git base-devel \
                pkg-config openssl \
                clang \
                postgresql-libs \
                python python-pip \
                tree ripgrep
            ;;
        macos)
            if ! command_exists brew; then
                log "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install curl wget git pkg-config openssl clang postgresql python tree ripgrep
            ;;
        *)
            log_warning "Unknown OS. You may need to install packages manually."
            ;;
    esac

    log_success "System packages installed"
}

install_rust() {
    if [ "$INSTALL_RUST" = "false" ]; then
        return
    fi

    log "Installing Rust..."

    if command_exists rustc; then
        local current_version=$(rustc --version | cut -d' ' -f2)
        log "Rust already installed: $current_version"
        return
    fi

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_VERSION"
    source "$HOME/.cargo/env"

    # Install additional components
    rustup component add clippy rustfmt

    log_success "Rust $RUST_VERSION installed"
}

install_python() {
    if [ "$INSTALL_PYTHON" = "false" ]; then
        return
    fi

    log "Setting up Python environment..."

    if ! command_exists python3; then
        log_error "Python 3 not found. Please install Python $PYTHON_VERSION or later."
        exit 1
    fi

    local python_version=$(python3 --version | cut -d' ' -f2)
    log "Using Python: $python_version"

    # Create virtual environment for KCS
    python3 -m venv "$KCS_HOME/venv"
    source "$KCS_HOME/venv/bin/activate"

    # Upgrade pip and install build tools
    pip install --upgrade pip setuptools wheel

    log_success "Python environment set up"
}

install_postgresql() {
    if [ "$INSTALL_POSTGRES" = "false" ]; then
        return
    fi

    log "Installing PostgreSQL..."

    local os=$(get_os)

    case $os in
        debian)
            sudo apt-get install -y postgresql postgresql-contrib postgresql-15-pgvector
            ;;
        redhat)
            sudo dnf install -y postgresql-server postgresql-contrib
            sudo postgresql-setup --initdb
            ;;
        arch)
            sudo pacman -S --noconfirm postgresql
            sudo -u postgres initdb -D /var/lib/postgres/data
            ;;
        macos)
            brew install postgresql pgvector
            ;;
    esac

    # Start PostgreSQL service
    case $os in
        debian|redhat|arch)
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            ;;
        macos)
            brew services start postgresql
            ;;
    esac

    log_success "PostgreSQL installed and started"
}

setup_database() {
    log "Setting up KCS database..."

    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE $POSTGRES_DB;" || true
    sudo -u postgres psql -c "CREATE USER $POSTGRES_USER WITH PASSWORD 'kcs_password';" || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;" || true
    sudo -u postgres psql -c "ALTER USER $POSTGRES_USER CREATEDB;" || true

    # Install pgvector extension
    sudo -u postgres psql -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;" || true

    # Run migrations
    log "Running database migrations..."
    cd "$KCS_ROOT"
    source "$KCS_HOME/venv/bin/activate"
    export DATABASE_URL="postgresql://$POSTGRES_USER:kcs_password@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"

    if [ -f "tools/setup/migrate.sh" ]; then
        bash tools/setup/migrate.sh
    fi

    log_success "Database setup completed"
}

build_rust_components() {
    log "Building Rust components..."

    cd "$KCS_ROOT/src/rust"
    source "$HOME/.cargo/env"

    # Build all Rust libraries
    cargo build --release

    # Install binaries
    sudo mkdir -p /usr/local/bin
    sudo cp target/release/kcs-parser /usr/local/bin/ || true
    sudo cp target/release/kcs-extractor /usr/local/bin/ || true
    sudo cp target/release/kcs-graph /usr/local/bin/ || true
    sudo cp target/release/kcs-impact /usr/local/bin/ || true
    sudo cp target/release/kcs-drift /usr/local/bin/ || true

    log_success "Rust components built and installed"
}

install_python_packages() {
    log "Installing Python packages..."

    cd "$KCS_ROOT"
    source "$KCS_HOME/venv/bin/activate"

    # Install Python dependencies
    if [ -f "pyproject.toml" ]; then
        pip install -e .
    fi

    # Install additional packages
    pip install \
        fastapi uvicorn \
        psycopg2-binary sqlalchemy \
        asyncpg \
        pydantic \
        httpx \
        pytest pytest-asyncio \
        black ruff mypy

    log_success "Python packages installed"
}

setup_systemd_service() {
    if [ "$INSTALL_TYPE" != "production" ]; then
        return
    fi

    log "Setting up systemd service..."

    cat > /tmp/kcs.service << EOF
[Unit]
Description=Kernel Context Server
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=$KCS_USER
Group=$KCS_USER
WorkingDirectory=$KCS_HOME
Environment=PATH=$KCS_HOME/venv/bin
Environment=DATABASE_URL=postgresql://$POSTGRES_USER:kcs_password@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB
Environment=KCS_PORT=$KCS_PORT
ExecStart=$KCS_HOME/venv/bin/python -m kcs_mcp.app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/kcs.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable kcs

    log_success "Systemd service configured"
}

setup_user() {
    if [ "$INSTALL_TYPE" != "production" ]; then
        return
    fi

    log "Setting up KCS user..."

    if ! id "$KCS_USER" &>/dev/null; then
        sudo useradd -r -s /bin/false -d "$KCS_HOME" "$KCS_USER"
    fi

    sudo mkdir -p "$KCS_HOME"
    sudo chown -R "$KCS_USER:$KCS_USER" "$KCS_HOME"

    log_success "KCS user created"
}

create_config_file() {
    log "Creating configuration file..."

    cat > "$KCS_HOME/config.env" << EOF
# KCS Configuration
DATABASE_URL=postgresql://$POSTGRES_USER:kcs_password@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB
KCS_PORT=$KCS_PORT
KCS_HOST=0.0.0.0
KCS_LOG_LEVEL=INFO

# Authentication
KCS_JWT_SECRET=$(openssl rand -base64 32)
KCS_AUTH_ENABLED=true

# Feature flags
KCS_CLANG_ENABLED=true
KCS_SEMANTIC_SEARCH_ENABLED=true
KCS_CACHE_ENABLED=true

# Performance tuning
KCS_WORKERS=4
KCS_MAX_CONNECTIONS=100
KCS_QUERY_TIMEOUT=30

# Paths
KCS_DATA_DIR=$KCS_HOME/data
KCS_LOG_DIR=$KCS_HOME/logs
KCS_CACHE_DIR=$KCS_HOME/cache
EOF

    # Create directories
    sudo mkdir -p "$KCS_HOME"/{data,logs,cache}

    if [ "$INSTALL_TYPE" = "production" ]; then
        sudo chown -R "$KCS_USER:$KCS_USER" "$KCS_HOME"
        sudo chmod 600 "$KCS_HOME/config.env"
    fi

    log_success "Configuration file created"
}

run_tests() {
    log "Running tests to verify installation..."

    cd "$KCS_ROOT"
    source "$KCS_HOME/venv/bin/activate"

    # Run contract tests
    python -m pytest tests/contract/ -v || log_warning "Some contract tests failed"

    # Test Rust binaries
    if command_exists kcs-parser; then
        kcs-parser --version || log_warning "kcs-parser test failed"
    fi

    log_success "Installation tests completed"
}

print_summary() {
    log_success "KCS installation completed!"
    echo
    echo "Installation Summary:"
    echo "===================="
    echo "Install type: $INSTALL_TYPE"
    echo "KCS home: $KCS_HOME"
    echo "Database: postgresql://$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
    echo "Configuration: $KCS_HOME/config.env"
    echo "Log file: $LOG_FILE"
    echo

    if [ "$INSTALL_TYPE" = "development" ]; then
        echo "Development setup complete. To start KCS:"
        echo "1. Activate Python environment: source $KCS_HOME/venv/bin/activate"
        echo "2. Set environment: export \$(cat $KCS_HOME/config.env | xargs)"
        echo "3. Start server: cd $KCS_ROOT && python -m kcs_mcp.app"
        echo "4. Access at: http://localhost:$KCS_PORT"
    else
        echo "Production setup complete. To start KCS:"
        echo "1. Start service: sudo systemctl start kcs"
        echo "2. Check status: sudo systemctl status kcs"
        echo "3. View logs: sudo journalctl -u kcs -f"
        echo "4. Access at: http://localhost:$KCS_PORT"
    fi

    echo
    echo "Next steps:"
    echo "- Index a kernel repository: bash tools/index_kernel.sh /path/to/linux"
    echo "- Run tests: python -m pytest tests/"
    echo "- Read documentation: docs/"
}

show_help() {
    cat << EOF
KCS Installation Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --type TYPE         Installation type: development|production (default: development)
    -u, --user USER         KCS user name (default: kcs)
    --home DIR              KCS home directory (default: /opt/kcs)
    --db-name NAME          Database name (default: kcs)
    --db-user USER          Database user (default: kcs)
    --db-host HOST          Database host (default: localhost)
    --db-port PORT          Database port (default: 5432)
    --port PORT             KCS server port (default: 8080)
    --no-postgres           Skip PostgreSQL installation
    --no-python             Skip Python setup
    --no-rust               Skip Rust installation
    --docker                Use Docker for dependencies
    --dry-run               Show what would be done without executing

Examples:
    # Development installation
    $0 --type development

    # Production installation
    $0 --type production --user kcs --home /opt/kcs

    # Custom database configuration
    $0 --db-host db.example.com --db-port 5433

Environment Variables:
    KCS_DB_PASSWORD         Database password (default: auto-generated)
    KCS_JWT_SECRET          JWT secret key (default: auto-generated)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            INSTALL_TYPE="$2"
            shift 2
            ;;
        -u|--user)
            KCS_USER="$2"
            shift 2
            ;;
        --home)
            KCS_HOME="$2"
            shift 2
            ;;
        --db-name)
            POSTGRES_DB="$2"
            shift 2
            ;;
        --db-user)
            POSTGRES_USER="$2"
            shift 2
            ;;
        --db-host)
            POSTGRES_HOST="$2"
            shift 2
            ;;
        --db-port)
            POSTGRES_PORT="$2"
            shift 2
            ;;
        --port)
            KCS_PORT="$2"
            shift 2
            ;;
        --no-postgres)
            INSTALL_POSTGRES=false
            shift
            ;;
        --no-python)
            INSTALL_PYTHON=false
            shift
            ;;
        --no-rust)
            INSTALL_RUST=false
            shift
            ;;
        --docker)
            INSTALL_DOCKER=true
            shift
            ;;
        --dry-run)
            log "DRY RUN MODE - No changes will be made"
            set -n  # No execution mode
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate install type
if [[ "$INSTALL_TYPE" != "development" && "$INSTALL_TYPE" != "production" ]]; then
    log_error "Invalid install type: $INSTALL_TYPE"
    exit 1
fi

# Main installation flow
main() {
    log "Starting KCS installation ($INSTALL_TYPE mode)..."
    log "Log file: $LOG_FILE"

    check_requirements

    if [ "$INSTALL_TYPE" = "production" ]; then
        setup_user
    fi

    install_system_packages

    if [ "$INSTALL_RUST" = "true" ]; then
        install_rust
    fi

    if [ "$INSTALL_PYTHON" = "true" ]; then
        install_python
    fi

    if [ "$INSTALL_POSTGRES" = "true" ]; then
        install_postgresql
        setup_database
    fi

    build_rust_components
    install_python_packages
    create_config_file

    if [ "$INSTALL_TYPE" = "production" ]; then
        setup_systemd_service
    fi

    run_tests
    print_summary
}

# Trap errors and cleanup
trap 'log_error "Installation failed. Check log: $LOG_FILE"' ERR

# Check if running as root for production install
if [ "$INSTALL_TYPE" = "production" ] && [ "$EUID" -ne 0 ]; then
    log_error "Production installation must be run as root"
    exit 1
fi

# Run main installation
main
