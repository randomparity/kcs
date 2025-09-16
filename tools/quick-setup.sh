#!/bin/bash
# KCS Quick Setup Script
#
# Automates the most common setup tasks for KCS development.
# Run this after cloning the repository to get started quickly.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KCS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
get_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."

    local missing=()

    # Check essential tools
    if ! command_exists git; then missing+=("git"); fi
    if ! command_exists curl; then missing+=("curl"); fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log "Please install them and run this script again."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

setup_environment() {
    log "Setting up environment configuration..."

    cd "$KCS_ROOT"

    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "Created .env from .env.example"
        else
            log_error ".env.example not found!"
            exit 1
        fi
    else
        log_warning ".env already exists, skipping"
    fi
}

setup_docker() {
    log "Setting up Docker environment..."

    if ! command_exists docker; then
        log_error "Docker not found. Please install Docker first:"
        local os=$(get_os)
        case $os in
            debian)
                echo "  sudo apt-get install docker.io docker-compose-v2"
                ;;
            redhat)
                echo "  sudo dnf install docker docker-compose"
                ;;
            macos)
                echo "  brew install docker docker-compose"
                ;;
        esac
        exit 1
    fi

    # Check Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker:"
        echo "  sudo systemctl start docker  # Linux"
        echo "  # or start Docker Desktop    # macOS"
        exit 1
    fi

    # Create data directories
    cd "$KCS_ROOT"
    if command_exists make; then
        make create-data-dirs
        log_success "Created data directories"
    else
        # Manual directory creation
        mkdir -p ./appdata/{postgres,redis,grafana,prometheus,kcs}/{data,logs}
        log_success "Created data directories manually"
    fi

    # Start services
    log "Starting Docker services..."
    docker compose --profile app up -d

    # Wait for services to be healthy
    log "Waiting for services to start..."
    sleep 10

    # Check service health
    if curl -s http://localhost:8080/health >/dev/null; then
        log_success "MCP server is running at http://localhost:8080"
    else
        log_warning "MCP server might still be starting. Check with:"
        echo "  docker compose logs kcs-mcp"
    fi
}

setup_rust() {
    log "Setting up Rust development environment..."

    if ! command_exists rustc; then
        log "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        rustup component add clippy rustfmt
        log_success "Rust installed"
    else
        log_success "Rust already installed: $(rustc --version)"
    fi

    # Build KCS tools
    cd "$KCS_ROOT"
    log "Building KCS analysis tools..."
    cargo build --release

    # Add to PATH suggestion
    log_success "KCS tools built successfully"
    log "To use KCS tools, add to your PATH:"
    echo "  export PATH=\"$KCS_ROOT/target/release:\$PATH\""
    echo "Or add this to your ~/.bashrc for persistence"
}

setup_local_development() {
    log "Setting up local development environment..."

    # Check Python
    if ! command_exists python3; then
        log_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi

    local python_version=$(python3 --version | cut -d' ' -f2)
    log "Using Python: $python_version"

    # Setup virtual environment
    cd "$KCS_ROOT"
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        log_success "Created Python virtual environment"
    fi

    # Activate and install
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e .
    log_success "Installed KCS Python packages"

    # Setup Rust
    setup_rust

    log_success "Local development environment ready!"
    log "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo "  export PATH=\"$KCS_ROOT/target/release:\$PATH\""
}

install_kernel_deps() {
    log "Installing kernel build dependencies..."

    local os=$(get_os)
    case $os in
        debian)
            log "Installing via apt..."
            echo "Run: sudo apt-get install -y flex bison build-essential libssl-dev libelf-dev"
            ;;
        redhat)
            log "Installing via dnf..."
            echo "Run: sudo dnf install -y flex bison gcc make openssl-devel elfutils-libelf-devel"
            ;;
        macos)
            log "Installing via brew..."
            echo "Run: brew install flex bison"
            ;;
        *)
            log_warning "Unknown OS. Please install flex, bison, and build tools manually"
            ;;
    esac
}

show_next_steps() {
    log_success "Setup complete! Next steps:"
    echo
    echo "🚀 Quick Start:"
    echo "  1. Test the API:"
    echo "     curl http://localhost:8080/health"
    echo
    echo "  2. Try a search query:"
    echo "     curl -X POST http://localhost:8080/mcp/tools/search_code \\"
    echo "       -H \"Content-Type: application/json\" \\"
    echo "       -H \"Authorization: Bearer dev-token\" \\"
    echo "       -d '{\"query\": \"hello world\", \"topK\": 3}'"
    echo
    echo "📚 Documentation:"
    echo "  - Complete guide: docs/INSTALLATION.md"
    echo "  - API reference: README.md"
    echo "  - Troubleshooting: README.md#troubleshooting"
    echo
    echo "🔧 Development:"
    echo "  - Run tests: make test"
    echo "  - Start dev server: make dev"
    echo "  - View logs: docker compose logs -f kcs-mcp"
    echo
    echo "🐧 Kernel Indexing:"
    echo "  - Install kernel deps (see output above)"
    echo "  - Clone Linux: git clone --depth 1 https://github.com/torvalds/linux.git ~/src/linux"
    echo "  - Index kernel: tools/index_kernel.sh ~/src/linux"
}

show_help() {
    cat << EOF
KCS Quick Setup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help
    --docker            Setup Docker environment (default)
    --local             Setup local development environment
    --rust-only         Only build Rust components
    --kernel-deps       Show kernel build dependency instructions
    --no-start          Don't start services automatically

Examples:
    $0                  # Setup Docker environment
    $0 --local          # Setup local development
    $0 --rust-only      # Just build Rust tools
    $0 --kernel-deps    # Show kernel dependency install commands

EOF
}

# Parse command line arguments
SETUP_TYPE="docker"
START_SERVICES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --docker)
            SETUP_TYPE="docker"
            shift
            ;;
        --local)
            SETUP_TYPE="local"
            shift
            ;;
        --rust-only)
            SETUP_TYPE="rust"
            shift
            ;;
        --kernel-deps)
            SETUP_TYPE="kernel-deps"
            shift
            ;;
        --no-start)
            START_SERVICES=false
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log "Starting KCS quick setup..."
    log "Setup type: $SETUP_TYPE"

    check_prerequisites
    setup_environment

    case $SETUP_TYPE in
        docker)
            if [ "$START_SERVICES" = true ]; then
                setup_docker
            else
                log "Docker setup prepared. Run 'docker compose --profile app up -d' to start."
            fi
            setup_rust
            install_kernel_deps
            ;;
        local)
            setup_local_development
            install_kernel_deps
            ;;
        rust)
            setup_rust
            ;;
        kernel-deps)
            install_kernel_deps
            exit 0
            ;;
    esac

    show_next_steps
}

# Run main function
main
