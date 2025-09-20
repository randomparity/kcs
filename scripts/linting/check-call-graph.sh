#!/bin/bash
# Linting script for call graph extraction modules
# Runs clippy and rustfmt checks specifically for call graph code

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "src/rust" ]]; then
    print_error "This script must be run from the KCS repository root"
    exit 1
fi

print_status "Running clippy and rustfmt checks for call graph modules..."

# Run clippy for kcs-parser with call graph focus
print_status "Checking kcs-parser call graph modules..."
cd src/rust/kcs-parser

if cargo clippy --all-targets --all-features -- -D warnings; then
    print_status "✓ kcs-parser clippy check passed"
else
    print_error "✗ kcs-parser clippy check failed"
    exit 1
fi

# Check rustfmt for kcs-parser
if cargo fmt --check; then
    print_status "✓ kcs-parser formatting check passed"
else
    print_warning "kcs-parser formatting needs adjustment, running rustfmt..."
    cargo fmt
    print_status "✓ kcs-parser formatting fixed"
fi

# Return to root
cd ../../..

# Run clippy for kcs-graph
print_status "Checking kcs-graph call graph modules..."
cd src/rust/kcs-graph

if cargo clippy --all-targets --all-features -- -D warnings; then
    print_status "✓ kcs-graph clippy check passed"
else
    print_error "✗ kcs-graph clippy check failed"
    exit 1
fi

# Check rustfmt for kcs-graph
if cargo fmt --check; then
    print_status "✓ kcs-graph formatting check passed"
else
    print_warning "kcs-graph formatting needs adjustment, running rustfmt..."
    cargo fmt
    print_status "✓ kcs-graph formatting fixed"
fi

# Return to root
cd ../../..

print_status "All call graph module checks completed successfully!"

# Optional: Run cargo check to ensure compilation
print_status "Running compilation check..."
if cargo check -p kcs-parser -p kcs-graph; then
    print_status "✓ Compilation check passed"
else
    print_error "✗ Compilation check failed"
    exit 1
fi

print_status "🎉 All call graph linting and formatting checks passed!"
