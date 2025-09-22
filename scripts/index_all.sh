#!/bin/bash
set -euo pipefail

# index_all.sh - Comprehensive Linux kernel indexing for KCS
# Purpose: Perform both symbol indexing and semantic search indexing
# Dependencies: KCS virtual environment, Linux kernel source

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KCS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[INDEX]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Configuration
KERNEL_PATH=""
SEMANTIC_MAX_FILES=1000
SKIP_SYMBOL_INDEX=false
SKIP_SEMANTIC_INDEX=false
VERBOSE=false

usage() {
    cat << EOF
Usage: $0 [OPTIONS] KERNEL_PATH

Comprehensive indexing of Linux kernel source for KCS.
Performs both symbol indexing and semantic search indexing.

ARGUMENTS:
    KERNEL_PATH             Path to Linux kernel source directory

OPTIONS:
    --semantic-max-files N  Maximum files for semantic indexing (default: 1000)
    --skip-symbol          Skip symbol/entry point indexing
    --skip-semantic        Skip semantic search indexing
    --verbose              Enable verbose output
    --help                 Show this help

INDEXING PROCESSES:
    1. Symbol Indexing:
       • Uses Rust-based KCS parser (tools/index_kernel.sh)
       • Extracts functions, symbols, entry points
       • Builds call graphs and dependencies
       • Populates symbol/entrypoint database tables
       • Enables exact symbol lookup queries

    2. Semantic Search Indexing:
       • Uses Python-based semantic search CLI
       • Creates text chunks with vector embeddings
       • Enables natural language queries
       • Populates indexed_content/vector_embedding tables
       • Supports similarity-based code search

EXAMPLES:
    $0 ~/src/linux                           # Full indexing (both symbol + semantic)
    $0 --skip-symbol ~/src/linux             # Semantic indexing only
    $0 --skip-semantic ~/src/linux           # Symbol indexing only
    $0 --semantic-max-files 500 ~/src/linux  # Limit semantic files

SEARCH EXAMPLES AFTER INDEXING:
    # Symbol search (exact matching)
    curl -X POST -H "Content-Type: application/json" \\
         -d '{"arguments": {"symbol": "kmalloc"}}' \\
         http://localhost:8080/mcp/tools/get_symbol

    # Semantic search (natural language)
    curl -X POST -H "Content-Type: application/json" \\
         -H "Authorization: Bearer dev-token" \\
         -d '{"query": "memory allocation error handling", "limit": 5}' \\
         http://localhost:8080/mcp/tools/semantic_search

EOF
}

validate_arguments() {
    if [[ -z "$KERNEL_PATH" ]]; then
        error "Kernel path is required. Use --help for usage information."
    fi

    if [[ ! -d "$KERNEL_PATH" ]]; then
        error "Kernel path does not exist: $KERNEL_PATH"
    fi

    if [[ ! -f "$KERNEL_PATH/Makefile" ]] || [[ ! -d "$KERNEL_PATH/kernel" ]]; then
        error "Invalid kernel source directory: $KERNEL_PATH (missing Makefile or kernel/ directory)"
    fi

    # Convert to absolute path
    KERNEL_PATH=$(cd "$KERNEL_PATH" && pwd)
    info "Using kernel source: $KERNEL_PATH"
}

validate_environment() {
    log "Validating indexing environment..."

    # Check required tools
    for tool in python3 jq find; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            error "Required tool not found: $tool"
        fi
    done

    # Check KCS virtual environment
    if [[ ! -f "$KCS_ROOT/.venv/bin/activate" ]]; then
        error "KCS virtual environment not found at $KCS_ROOT/.venv. Run 'make install' first."
    fi

    # Check indexing tools exist
    if [[ "$SKIP_SYMBOL_INDEX" != "true" ]] && [[ ! -f "$KCS_ROOT/tools/index_kernel.sh" ]]; then
        error "Symbol indexing tool not found at $KCS_ROOT/tools/index_kernel.sh"
    fi

    # Check if KCS services are running (for verification)
    if ! curl -s http://localhost:8080/health >/dev/null 2>&1; then
        warn "KCS server not accessible at http://localhost:8080 - verification will be skipped"
        warn "Start KCS with: make docker-compose-up-app"
    fi

    info "Environment validation passed"
}

run_symbol_indexing() {
    if [[ "$SKIP_SYMBOL_INDEX" == "true" ]]; then
        info "Skipping symbol indexing (--skip-symbol specified)"
        return 0
    fi

    log "Starting symbol indexing (traditional KCS)..."

    # Activate virtual environment
    source "$KCS_ROOT/.venv/bin/activate"

    # Run symbol indexing using existing tool
    cd "$KCS_ROOT"
    info "Running symbol indexing with tools/index_kernel.sh..."

    local index_start_time=$(date +%s)
    local log_args=""

    if [[ "$VERBOSE" == "true" ]]; then
        log_args="2>&1 | tee /tmp/kcs_symbol_index.log"
    else
        log_args=">/tmp/kcs_symbol_index.log 2>&1"
    fi

    if eval "DATABASE_URL=\"postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs\" \
             ./tools/index_kernel.sh \
             --config x86_64:defconfig \
             --chunk-size 32MB \
             \"$KERNEL_PATH\" $log_args"; then

        local index_end_time=$(date +%s)
        local duration=$((index_end_time - index_start_time))
        log "✅ Symbol indexing completed successfully in ${duration}s"
    else
        error "❌ Symbol indexing failed. Check /tmp/kcs_symbol_index.log for details"
    fi
}

create_semantic_indexing_script() {
    local script_path=$(mktemp)

    cat > "$script_path" << 'EOF'
#!/usr/bin/env python3
"""
Semantic search indexing for Linux kernel source.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add semantic search to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from semantic_search.cli.index_commands import index_directory
from semantic_search.database.connection import DatabaseConfig, init_database_connection

async def main():
    max_files = int(sys.argv[1])
    kernel_path = sys.argv[2]
    verbose = sys.argv[3] == "true"

    print(f"🔄 Starting semantic search indexing...")
    print(f"   Kernel path: {kernel_path}")
    print(f"   Max files: {max_files}")

    start_time = time.time()

    try:
        # Initialize database connection
        db_config = DatabaseConfig.from_env()
        await init_database_connection(db_config)

        # Define file patterns for kernel source
        file_patterns = ["*.c", "*.h", "*.S"]
        exclude_patterns = [
            "*/Documentation/*",
            "*/tools/*",
            "*/scripts/*",
            "*/samples/*",
            "*/arch/arm/*",
            "*/arch/arm64/*",
            "*/arch/mips/*",
            "*/arch/powerpc/*",
            "*/arch/s390/*",
            "*/arch/sparc/*",
            "*/drivers/gpu/*",
            "*/drivers/media/*",
            "*/drivers/staging/*",
        ]

        if verbose:
            print(f"   File patterns: {file_patterns}")
            print(f"   Exclude patterns: {exclude_patterns[:3]}... ({len(exclude_patterns)} total)")

        # Index the kernel directory
        result = await index_directory(
            directory_path=kernel_path,
            recursive=True,
            file_patterns=file_patterns,
            exclude_patterns=exclude_patterns,
            max_files=max_files,
            force_reindex=False
        )

        end_time = time.time()
        duration = end_time - start_time

        # Print results
        print(f"\n📊 Semantic Indexing Results:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Files processed: {result.get('files_indexed', 0)}")
        print(f"   Files failed: {result.get('files_failed', 0)}")
        print(f"   Duration: {duration:.1f}s")

        if result.get('status') == 'completed':
            print(f"🎉 Semantic indexing completed successfully!")
        else:
            print(f"⚠️  Semantic indexing completed with issues")
            if verbose and 'errors' in result:
                for error in result['errors'][:5]:  # Show first 5 errors
                    print(f"     Error: {error}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Semantic indexing failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

    echo "$script_path"
}

run_semantic_indexing() {
    if [[ "$SKIP_SEMANTIC_INDEX" == "true" ]]; then
        info "Skipping semantic indexing (--skip-semantic specified)"
        return 0
    fi

    log "Starting semantic search indexing..."

    # Activate virtual environment
    source "$KCS_ROOT/.venv/bin/activate"

    # Create semantic indexing script
    local indexing_script=$(create_semantic_indexing_script)
    chmod +x "$indexing_script"

    # Set environment variables
    export PYTHONPATH="$KCS_ROOT/src/python:${PYTHONPATH:-}"
    export DATABASE_URL="postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs"

    # Run semantic indexing
    cd "$KCS_ROOT"
    if python3 "$indexing_script" "$SEMANTIC_MAX_FILES" "$KERNEL_PATH" "$VERBOSE"; then
        log "✅ Semantic indexing completed successfully"
    else
        error "❌ Semantic indexing failed"
    fi

    # Clean up
    rm -f "$indexing_script"
}

verify_symbol_indexing() {
    if [[ "$SKIP_SYMBOL_INDEX" == "true" ]]; then
        return 0
    fi

    info "Verifying symbol indexing results..."

    # Check if psql is available
    if ! command -v psql >/dev/null 2>&1; then
        warn "psql not available - skipping symbol verification"
        return 0
    fi

    # Check database for symbols
    local symbol_count
    local entry_point_count

    symbol_count=$(PGPASSWORD=kcs_dev_password_change_in_production \
                   psql -h localhost -U kcs -d kcs -t -c "SELECT COUNT(*) FROM symbol;" 2>/dev/null | xargs || echo "0")

    entry_point_count=$(PGPASSWORD=kcs_dev_password_change_in_production \
                        psql -h localhost -U kcs -d kcs -t -c "SELECT COUNT(*) FROM entrypoint;" 2>/dev/null | xargs || echo "0")

    if [[ $symbol_count -gt 1000 ]]; then
        info "✅ Symbol indexing verified: $symbol_count symbols, $entry_point_count entry points"
    else
        warn "⚠️  Symbol indexing may be incomplete: only $symbol_count symbols found"
    fi
}

verify_semantic_indexing() {
    if [[ "$SKIP_SEMANTIC_INDEX" == "true" ]]; then
        return 0
    fi

    info "Verifying semantic search indexing..."

    # Test semantic search if server is available
    if ! curl -s http://localhost:8080/health >/dev/null 2>&1; then
        warn "KCS server not available - skipping semantic search verification"
        return 0
    fi

    # Test semantic search
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer dev-token" \
        "http://localhost:8080/mcp/tools/semantic_search" \
        -d '{"query": "memory allocation", "limit": 3}' \
        --max-time 10 \
        2>/dev/null)

    if echo "$response" | jq -e '.results' >/dev/null 2>&1; then
        local result_count=$(echo "$response" | jq -r '.results | length')
        local total_results=$(echo "$response" | jq -r '.total_results // 0')

        if [[ $result_count -gt 0 ]]; then
            info "✅ Semantic search verified: $result_count/$total_results results"
            if [[ "$VERBOSE" == "true" ]]; then
                echo "Sample result:"
                echo "$response" | jq -r '.results[0] | "  File: \(.file_path // "unknown"):\(.line_start // 0)"'
            fi
        else
            warn "⚠️  No semantic search results found"
        fi
    else
        warn "⚠️  Semantic search verification failed"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "Response: $response"
        fi
    fi
}

print_summary() {
    echo
    log "🎉 Comprehensive kernel indexing complete!"
    echo
    info "Indexing Summary:"

    if [[ "$SKIP_SYMBOL_INDEX" != "true" ]]; then
        echo "  ✅ Symbol indexing: Functions, entry points, call graphs"
    else
        echo "  ⏭️  Symbol indexing: Skipped"
    fi

    if [[ "$SKIP_SEMANTIC_INDEX" != "true" ]]; then
        echo "  ✅ Semantic indexing: Text chunks, vector embeddings"
    else
        echo "  ⏭️  Semantic indexing: Skipped"
    fi

    echo
    info "Available search types:"
    if [[ "$SKIP_SYMBOL_INDEX" != "true" ]]; then
        echo "  • Symbol search: curl -X POST -d '{\"arguments\": {\"symbol\": \"kmalloc\"}}' http://localhost:8080/mcp/tools/get_symbol"
    fi
    if [[ "$SKIP_SEMANTIC_INDEX" != "true" ]]; then
        echo "  • Semantic search: curl -X POST -H \"Authorization: Bearer dev-token\" -d '{\"query\": \"memory allocation\", \"limit\": 5}' http://localhost:8080/mcp/tools/semantic_search"
    fi
    echo
    info "Log files:"
    echo "  • Symbol indexing: /tmp/kcs_symbol_index.log"
    echo "  • This script output can be logged with: $0 $* 2>&1 | tee indexing.log"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                usage
                exit 0
                ;;
            --semantic-max-files)
                SEMANTIC_MAX_FILES="$2"
                shift 2
                ;;
            --skip-symbol)
                SKIP_SYMBOL_INDEX=true
                shift
                ;;
            --skip-semantic)
                SKIP_SEMANTIC_INDEX=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -*)
                error "Unknown option: $1"
                ;;
            *)
                if [[ -z "$KERNEL_PATH" ]]; then
                    KERNEL_PATH="$1"
                else
                    error "Multiple kernel paths specified: $KERNEL_PATH and $1"
                fi
                shift
                ;;
        esac
    done
}

main() {
    parse_args "$@"
    validate_arguments
    validate_environment

    echo "KCS Comprehensive Kernel Indexing"
    echo "================================="
    echo
    info "Kernel source: $KERNEL_PATH"
    info "Symbol indexing: $([ "$SKIP_SYMBOL_INDEX" == "true" ] && echo "disabled" || echo "enabled")"
    info "Semantic indexing: $([ "$SKIP_SEMANTIC_INDEX" == "true" ] && echo "disabled" || echo "enabled (max $SEMANTIC_MAX_FILES files)")"
    echo

    # Run both indexing processes
    run_symbol_indexing
    echo
    run_semantic_indexing
    echo

    # Verify both processes worked
    verify_symbol_indexing
    verify_semantic_indexing

    print_summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
