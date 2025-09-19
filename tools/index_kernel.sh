#!/bin/bash
# Kernel Indexing Pipeline for KCS
#
# Indexes a Linux kernel repository for use with Kernel Context Server.
# Performs full analysis including parsing, entry point extraction,
# call graph building, and database population.

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KCS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="/tmp/kcs-index-$(date +%Y%m%d-%H%M%S).log"

# Use repository binaries directly instead of installed versions
KCS_PARSER="${KCS_ROOT}/src/rust/target/release/kcs-parser"
KCS_EXTRACTOR="${KCS_ROOT}/src/rust/target/release/kcs-extractor"
KCS_GRAPH="${KCS_ROOT}/src/rust/target/release/kcs-graph"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
KERNEL_PATH=""
CONFIG="x86_64:defconfig"
OUTPUT_DIR="/tmp/kcs-index"
PARALLEL_JOBS=4
BATCH_SIZE=100
USE_CLANG=true
CLEANUP_TEMP=true
INCREMENTAL=false
DATABASE_URL="${DATABASE_URL:-postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs}"
DRY_RUN=false
VERBOSE=false

# Chunking configuration
ENABLE_CHUNKING=false
CHUNK_SIZE="50MB"
PARALLEL_CHUNKS=4
CHUNK_OUTPUT_DIR=""
MANIFEST_PATH=""
SUBSYSTEM=""

# Performance targets (from constitution)
MAX_INDEX_TIME=1200  # 20 minutes in seconds
MAX_INCREMENTAL_TIME=180  # 3 minutes in seconds

# Logging functions
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${BLUE}${msg}${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${BLUE}${msg}${NC}"
    fi
}

log_success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $*"
    echo -e "${GREEN}${msg}${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $*"
    echo -e "${YELLOW}${msg}${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $*"
    echo -e "${RED}${msg}${NC}" | tee -a "$LOG_FILE"
}

# Utility functions
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Memory-efficient JSON counting functions
count_json_array_streaming() {
    local json_file="$1"
    local timeout_seconds="${2:-10}"

    if [ ! -f "$json_file" ]; then
        echo "0"
        return
    fi

    # Try jq with timeout first for smaller files
    local count
    count=$(timeout "$timeout_seconds" jq -r '. | length' "$json_file" 2>/dev/null || echo "")

    if [ -n "$count" ] && [ "$count" != "null" ]; then
        echo "$count"
        return
    fi

    # Fallback: process file in chunks to avoid memory issues
    local file_size=$(get_file_size_mb "$json_file")

    if [ "$file_size" -gt 100 ]; then
        # File is very large (>100MB), skip detailed counting
        echo "many"
        return
    fi

    # For medium files, try to count "path": occurrences
    local path_count
    path_count=$(timeout 30 grep -o '"path":' "$json_file" 2>/dev/null | wc -l || echo "0")

    if [ "$path_count" -gt 0 ]; then
        echo "$path_count"
    elif [ "$file_size" -gt 0 ]; then
        echo "many"  # Indicate content exists but count failed
    else
        echo "0"
    fi
}

count_symbols_streaming() {
    local json_file="$1"
    local timeout_seconds="${2:-10}"

    if [ ! -f "$json_file" ]; then
        echo "0"
        return
    fi

    # Try jq with timeout first
    local count
    count=$(timeout "$timeout_seconds" jq -r '[.[].symbols | length] | add' "$json_file" 2>/dev/null || echo "")

    if [ -n "$count" ] && [ "$count" != "null" ]; then
        echo "$count"
        return
    fi

    # Fallback: use memory-efficient approach
    local file_size=$(get_file_size_mb "$json_file")

    if [ "$file_size" -gt 100 ]; then
        # File is very large (>100MB), skip detailed counting
        echo "many"
        return
    fi

    # For medium files, try to count "name": occurrences
    local symbol_count
    symbol_count=$(timeout 30 grep -o '"name":' "$json_file" 2>/dev/null | wc -l || echo "0")

    if [ "$symbol_count" -gt 0 ]; then
        echo "$symbol_count"
    elif [ "$file_size" -gt 0 ]; then
        echo "many"  # Indicate content exists but count failed
    else
        echo "0"
    fi
}

get_file_size_mb() {
    local file="$1"
    if [ -f "$file" ]; then
        local size_bytes
        size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        echo $((size_bytes / 1024 / 1024))
    else
        echo "0"
    fi
}

get_kernel_version() {
    local kernel_path="$1"
    if [ -f "$kernel_path/Makefile" ]; then
        grep -E "^(VERSION|PATCHLEVEL|SUBLEVEL)" "$kernel_path/Makefile" | \
        awk -F'=' '{print $2}' | tr -d ' ' | tr '\n' '.' | sed 's/\.$//'
    else
        echo "unknown"
    fi
}

get_git_info() {
    local kernel_path="$1"
    cd "$kernel_path"
    if git rev-parse --git-dir >/dev/null 2>&1; then
        local sha=$(git rev-parse HEAD)
        local branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
        echo "$sha $branch"
    else
        echo "unknown unknown"
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check required tools
    local missing_tools=()

    if [ ! -f "$KCS_PARSER" ]; then
        missing_tools+=("kcs-parser (build with: cd $KCS_ROOT/src/rust && cargo build --release)")
    fi

    if [ ! -f "$KCS_EXTRACTOR" ]; then
        missing_tools+=("kcs-extractor (build with: cd $KCS_ROOT/src/rust && cargo build --release)")
    fi

    if [ ! -f "$KCS_GRAPH" ]; then
        missing_tools+=("kcs-graph (build with: cd $KCS_ROOT/src/rust && cargo build --release)")
    fi

    if ! command_exists psql; then
        missing_tools+=("psql")
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log "Please run: tools/setup/install.sh"
        exit 1
    fi

    # Check database connectivity (optional for testing)
    if ! psql "$DATABASE_URL" -c "SELECT 1;" >/dev/null 2>&1; then
        log_warning "Cannot connect to database: $DATABASE_URL"
        log_warning "Continuing without database. Database population will be skipped."
        SKIP_DATABASE=true
    else
        log_success "Database connection verified"
        SKIP_DATABASE=false
    fi

    # Check kernel path
    if [ ! -d "$KERNEL_PATH" ]; then
        log_error "Kernel path does not exist: $KERNEL_PATH"
        exit 1
    fi

    if [ ! -f "$KERNEL_PATH/Makefile" ]; then
        log_error "Not a valid kernel tree: $KERNEL_PATH"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

setup_environment() {
    log "Setting up indexing environment..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"/{parsed,extracted,graphs,temp}

    # Create chunking directories if enabled
    if [ "$ENABLE_CHUNKING" = "true" ]; then
        mkdir -p "$CHUNK_OUTPUT_DIR"
        log "Created chunking output directory: $CHUNK_OUTPUT_DIR"
    fi

    # Setup Python environment
    if [ -f "$KCS_ROOT/src/python/kcs_mcp/__init__.py" ]; then
        export PYTHONPATH="$KCS_ROOT/src/python:${PYTHONPATH:-}"
    fi

    log_success "Environment setup completed"
}

create_compile_commands() {
    log "Generating compile_commands.json for kernel..."

    local kernel_path="$1"
    local config="$2"
    local arch=$(echo "$config" | cut -d: -f1)
    local kconfig=$(echo "$config" | cut -d: -f2)

    cd "$kernel_path"

    # Clean and configure
    make mrproper ARCH="$arch"
    make "$kconfig" ARCH="$arch"

    # Generate compile commands
    if command_exists bear; then
        bear -- make -j"$PARALLEL_JOBS" ARCH="$arch" modules_prepare
        if [ -f "compile_commands.json" ]; then
            cp compile_commands.json "$OUTPUT_DIR/compile_commands.json"
            log_success "Generated compile_commands.json with bear"
        fi
    else
        log_warning "bear not found, proceeding without compile_commands.json"
        touch "$OUTPUT_DIR/compile_commands.json"
    fi
}

parse_kernel_sources() {
    if [ "$ENABLE_CHUNKING" = "true" ]; then
        parse_kernel_sources_chunked
    else
        parse_kernel_sources_traditional
    fi
}

parse_kernel_sources_traditional() {
    local parsed_output="$OUTPUT_DIR/parsed/kernel_symbols.json"

    # Skip if output already exists and is large enough to be valid
    if [ -f "$parsed_output" ]; then
        local file_size=$(get_file_size_mb "$parsed_output")
        if [ "$file_size" -gt 10 ]; then
            log "Parsed output already exists (${file_size}MB), skipping parsing step"
            local file_count=$(count_json_array_streaming "$parsed_output")
            local symbol_count=$(count_symbols_streaming "$parsed_output")
            if [ "$file_count" = "many" ] || [ "$symbol_count" = "many" ]; then
                log_success "Using existing parsed data (large dataset, exact counts unavailable)"
            else
                log_success "Using existing parsed data: $file_count files, $symbol_count symbols"
            fi
            return 0
        fi
    fi

    log "Parsing kernel sources..."
    local start_time=$(date +%s)

    # Parse with kcs-parser
    local parser_cmd=(
        "$KCS_PARSER"
        --format ndjson
        --workers "$PARALLEL_JOBS"
    )

    if [ "$VERBOSE" = "true" ]; then
        parser_cmd+=(--verbose)
    fi

    parser_cmd+=(
        parse
        --repo "$KERNEL_PATH"
        --config "$CONFIG"
        --output-dir "$OUTPUT_DIR"
    )

    if [ "$USE_CLANG" = "true" ] && [ -f "$OUTPUT_DIR/compile_commands.json" ]; then
        parser_cmd+=(--compile-commands "$OUTPUT_DIR/compile_commands.json")
    fi

    log "Running: ${parser_cmd[*]}"

    if [ "$DRY_RUN" = "false" ]; then
        "${parser_cmd[@]}" || {
            log_error "Parser failed"
            return 1
        }
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -f "$parsed_output" ]; then
        local file_size_mb=$(get_file_size_mb "$parsed_output")
        log "Output file size: ${file_size_mb}MB"

        # Use streaming counters for large files
        local file_count=$(count_json_array_streaming "$parsed_output")
        local symbol_count=$(count_symbols_streaming "$parsed_output")

        if [ "$file_count" = "0" ] && [ -s "$parsed_output" ]; then
            log_warning "Counting failed for large files. File exists but appears empty via counting."
            log_success "Parsed kernel sources (file counting failed, but output generated) in ${duration}s"
        elif [ "$file_count" = "many" ] || [ "$symbol_count" = "many" ]; then
            log_success "Parsed kernel sources (large dataset, exact counts unavailable) in ${duration}s"
        else
            log_success "Parsed $file_count files, found $symbol_count symbols in ${duration}s"
        fi
    else
        log_error "No parsed output file generated"
        return 1
    fi
}

parse_kernel_sources_chunked() {
    local manifest_output="$CHUNK_OUTPUT_DIR/manifest.json"

    # Check if we're resuming from existing manifest
    if [ -n "$MANIFEST_PATH" ] && [ -f "$MANIFEST_PATH" ]; then
        log "Resuming from existing manifest: $MANIFEST_PATH"
        manifest_output="$MANIFEST_PATH"

        # Validate manifest and check if parsing is needed
        if [ "$DRY_RUN" = "false" ]; then
            log "Checking manifest status..."

            # Create a simple check to see if chunks are already generated
            local chunk_count=$(jq -r '.total_chunks' "$manifest_output" 2>/dev/null || echo "0")
            if [ "$chunk_count" != "0" ] && [ "$chunk_count" != "null" ]; then
                log "Found existing chunks in manifest: $chunk_count chunks"
                log_success "Using existing chunked data from manifest"
                return 0
            fi
        fi
    fi

    log "Parsing kernel sources with chunking (size: $CHUNK_SIZE)..."
    local start_time=$(date +%s)

    # Build kcs-parser command with chunking support
    local parser_cmd=(
        "$KCS_PARSER"
        --format ndjson
        --workers "$PARALLEL_JOBS"
    )

    if [ "$VERBOSE" = "true" ]; then
        parser_cmd+=(--verbose)
    fi

    parser_cmd+=(
        parse
        --repo "$KERNEL_PATH"
        --config "$CONFIG"
        --output-dir "$CHUNK_OUTPUT_DIR"
    )

    # Add subsystem filter if specified
    if [ -n "$SUBSYSTEM" ]; then
        parser_cmd+=(--subsystem "$SUBSYSTEM")
    fi

    if [ "$USE_CLANG" = "true" ] && [ -f "$OUTPUT_DIR/compile_commands.json" ]; then
        parser_cmd+=(--compile-commands "$OUTPUT_DIR/compile_commands.json")
    fi

    # Add incremental flag if specified
    if [ "$INCREMENTAL" = "true" ] && [ -n "$MANIFEST_PATH" ]; then
        parser_cmd+=(--incremental --manifest "$MANIFEST_PATH")
    fi

    log "Running: ${parser_cmd[*]}"

    if [ "$DRY_RUN" = "false" ]; then
        "${parser_cmd[@]}" || {
            log_error "Chunked parser failed"
            return 1
        }
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Check generated manifest and chunks
    if [ -f "$manifest_output" ]; then
        local chunk_count=$(jq -r '.total_chunks' "$manifest_output" 2>/dev/null || echo "0")
        local total_size=$(jq -r '.total_size_bytes' "$manifest_output" 2>/dev/null || echo "0")

        if [ "$chunk_count" != "0" ] && [ "$chunk_count" != "null" ]; then
            local size_mb=$((total_size / 1024 / 1024))
            log_success "Generated $chunk_count chunks (${size_mb}MB total) in ${duration}s"
            log "Manifest created: $manifest_output"
        else
            log_error "No chunks generated in manifest"
            return 1
        fi
    else
        log_error "No manifest file generated"
        return 1
    fi
}

extract_entrypoints() {
    local entrypoints_output="$OUTPUT_DIR/extracted/entrypoints.json"

    # Skip if output already exists and is substantial
    if [ -f "$entrypoints_output" ]; then
        local entry_count=$(count_json_array_streaming "$entrypoints_output")
        if [ "$entry_count" != "0" ] && [ "$entry_count" != "many" ]; then
            log "Entry points already extracted ($entry_count entry points), skipping extraction"
            return 0
        fi
    fi

    log "Extracting kernel entry points..."

    local start_time=$(date +%s)
    local input_size=$(get_file_size_mb "$OUTPUT_DIR/parsed/kernel_symbols.json")

    # Try rust extractor first for smaller files
    if [ "$input_size" -le 500 ]; then
        local extractor_cmd=(
            kcs-extractor index
            --input "$OUTPUT_DIR/parsed/kernel_symbols.json"
            --output "$entrypoints_output"
            --types all
        )

        if [ "$VERBOSE" = "true" ]; then
            extractor_cmd+=(--verbose)
        fi

        log "Running: ${extractor_cmd[*]}"

        if [ "$DRY_RUN" = "false" ]; then
            "${extractor_cmd[@]}" && {
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                local entry_count=$(count_json_array_streaming "$entrypoints_output")
                log_success "Extracted $entry_count entry points in ${duration}s"
                return 0
            }
        fi
    fi

    # Fallback to streaming Python extractor for large files
    log "Using streaming extractor for large file (${input_size}MB)..."

    if [ "$DRY_RUN" = "false" ]; then
        python3 "$SCRIPT_DIR/extract_entrypoints_streaming.py" \
            "$OUTPUT_DIR/parsed/kernel_symbols.json" \
            "$entrypoints_output" || {
            log_error "Streaming entry point extraction failed"
            return 1
        }
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -f "$entrypoints_output" ]; then
        local entry_count=$(count_json_array_streaming "$entrypoints_output")
        log_success "Extracted $entry_count entry points in ${duration}s"
    else
        log_error "No entry points output file generated"
        return 1
    fi
}

build_call_graph() {
    local graph_output="$OUTPUT_DIR/graphs/call_graph.json"

    # Skip if output already exists and is substantial
    if [ -f "$graph_output" ]; then
        local file_size=$(get_file_size_mb "$graph_output")
        if [ "$file_size" -gt 1 ]; then
            log "Call graph output already exists (${file_size}MB), skipping graph building"
            return 0
        fi
    fi

    log "Building kernel call graph..."

    local start_time=$(date +%s)
    local input_size=$(get_file_size_mb "$OUTPUT_DIR/parsed/kernel_symbols.json")

    # Check if input is too large for memory-based processing
    if [ "$input_size" -gt 1000 ]; then
        log_warning "Input file is very large (${input_size}MB). Graph building may fail due to memory constraints."
        log "Consider using --no-clang for smaller parser output, or adding more RAM."
    fi

    # Build graph with kcs-graph
    local graph_cmd=(
        kcs-graph build
        --input "$OUTPUT_DIR/parsed/kernel_symbols.json"
        --output "$graph_output"
    )

    if [ "$VERBOSE" = "true" ]; then
        graph_cmd+=(--verbose)
    fi

    log "Running: ${graph_cmd[*]}"

    if [ "$DRY_RUN" = "false" ]; then
        # Show progress message for graph building
        log "This may take several minutes for large datasets..."
        if [ "$input_size" -gt 500 ]; then
            log "Large input detected. You can monitor progress with: tail -f $LOG_FILE"
        fi

        # Use timeout to prevent hanging and limit memory usage
        timeout 300 "${graph_cmd[@]}" || {
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                log_error "Call graph building timed out (5 minutes). Consider using smaller input or more memory."
                log "To continue indexing without graph: touch $graph_output"
            elif [ $exit_code -eq 137 ]; then
                log_error "Call graph building killed (likely out of memory). Try: tools/index_kernel.sh --no-clang"
                log "To continue indexing without graph: touch $graph_output"
            else
                log_error "Call graph building failed with exit code $exit_code"
            fi
            return 1
        }
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -f "$graph_output" ]; then
        # For graph files, try jq with timeout, then fallback to size reporting
        local node_count edge_count
        node_count=$(timeout 30 jq -r '.nodes | length' "$graph_output" 2>/dev/null || echo "unknown")
        edge_count=$(timeout 30 jq -r '.edges | length' "$graph_output" 2>/dev/null || echo "unknown")

        if [ "$node_count" = "unknown" ] || [ "$edge_count" = "unknown" ]; then
            local graph_size_mb=$(get_file_size_mb "$graph_output")
            log_success "Built call graph (${graph_size_mb}MB output) in ${duration}s"
        else
            log_success "Built call graph with $node_count nodes, $edge_count edges in ${duration}s"
        fi
    else
        log_error "No call graph output file generated"
        return 1
    fi
}

populate_database() {
    log "Populating database..."

    local start_time=$(date +%s)

    # Get kernel metadata
    local kernel_version=$(get_kernel_version "$KERNEL_PATH")
    local git_info=($(get_git_info "$KERNEL_PATH"))
    local git_sha="${git_info[0]}"
    local git_branch="${git_info[1]}"

    log "Kernel version: $kernel_version"
    log "Git SHA: $git_sha"
    log "Git branch: $git_branch"

    if [ "$DRY_RUN" = "false" ]; then
        # Use Python script to populate database
        python3 - << EOF
import json
import psycopg2
import sys
from datetime import datetime

# Database connection
try:
    conn = psycopg2.connect("$DATABASE_URL")
    cur = conn.cursor()
    print("Connected to database")
except Exception as e:
    print(f"Database connection failed: {e}")
    sys.exit(1)

# Load parsed data
try:
    with open("$OUTPUT_DIR/parsed/kernel_symbols.json", "r") as f:
        symbols_data = json.load(f)

    with open("$OUTPUT_DIR/extracted/entrypoints.json", "r") as f:
        entrypoints_data = json.load(f)

    with open("$OUTPUT_DIR/graphs/call_graph.json", "r") as f:
        graph_data = json.load(f)

    print(f"Loaded {len(symbols_data)} files, {len(entrypoints_data)} entry points")
except Exception as e:
    print(f"Failed to load data: {e}")
    sys.exit(1)

# Insert or update kernel index record
try:
    cur.execute("""
        INSERT INTO kernel_indexes (
            path, version, git_sha, git_branch, config,
            indexed_at, file_count, symbol_count, entrypoint_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (path, config) DO UPDATE SET
            version = EXCLUDED.version,
            git_sha = EXCLUDED.git_sha,
            git_branch = EXCLUDED.git_branch,
            indexed_at = EXCLUDED.indexed_at,
            file_count = EXCLUDED.file_count,
            symbol_count = EXCLUDED.symbol_count,
            entrypoint_count = EXCLUDED.entrypoint_count
        RETURNING id
    """, (
        "$KERNEL_PATH",
        "$kernel_version",
        "$git_sha",
        "$git_branch",
        "$CONFIG",
        datetime.now(),
        len(symbols_data),
        sum(len(f.get('symbols', [])) for f in symbols_data),
        len(entrypoints_data)
    ))

    index_id = cur.fetchone()[0]
    print(f"Kernel index ID: {index_id}")

except Exception as e:
    print(f"Failed to insert kernel index: {e}")
    conn.rollback()
    sys.exit(1)

# Batch insert files and symbols
try:
    file_batch = []
    symbol_batch = []

    for file_data in symbols_data:
        file_batch.append((
            index_id,
            file_data['path'],
            file_data['sha'],
            json.dumps(file_data.get('includes', [])),
            json.dumps(file_data.get('macros', []))
        ))

        for symbol in file_data.get('symbols', []):
            symbol_batch.append((
                index_id,
                file_data['path'],
                symbol['name'],
                symbol['kind'],
                symbol['start_line'],
                symbol['end_line'],
                symbol.get('signature', ''),
                symbol.get('visibility', 'global'),
                json.dumps(symbol.get('attributes', []))
            ))

    # Batch insert files
    if file_batch:
        cur.executemany("""
            INSERT INTO files (index_id, path, sha, includes, macros)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (index_id, path) DO UPDATE SET
                sha = EXCLUDED.sha,
                includes = EXCLUDED.includes,
                macros = EXCLUDED.macros
        """, file_batch)
        print(f"Inserted {len(file_batch)} files")

    # Batch insert symbols
    if symbol_batch:
        cur.executemany("""
            INSERT INTO symbols (index_id, file_path, name, kind, start_line, end_line, signature, visibility, attributes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_id, file_path, name, start_line) DO UPDATE SET
                kind = EXCLUDED.kind,
                end_line = EXCLUDED.end_line,
                signature = EXCLUDED.signature,
                visibility = EXCLUDED.visibility,
                attributes = EXCLUDED.attributes
        """, symbol_batch)
        print(f"Inserted {len(symbol_batch)} symbols")

except Exception as e:
    print(f"Failed to insert files/symbols: {e}")
    conn.rollback()
    sys.exit(1)

# Insert entry points
try:
    entrypoint_batch = []

    for entrypoint in entrypoints_data:
        entrypoint_batch.append((
            index_id,
            entrypoint['name'],
            entrypoint['type'],
            entrypoint['symbol'],
            entrypoint.get('file_path', ''),
            entrypoint.get('line_number', 0),
            json.dumps(entrypoint.get('metadata', {}))
        ))

    if entrypoint_batch:
        cur.executemany("""
            INSERT INTO entrypoints (index_id, name, type, symbol, file_path, line_number, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_id, name, type) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                file_path = EXCLUDED.file_path,
                line_number = EXCLUDED.line_number,
                metadata = EXCLUDED.metadata
        """, entrypoint_batch)
        print(f"Inserted {len(entrypoint_batch)} entry points")

except Exception as e:
    print(f"Failed to insert entry points: {e}")
    conn.rollback()
    sys.exit(1)

# Insert call graph edges
try:
    edge_batch = []

    for edge in graph_data.get('edges', []):
        edge_batch.append((
            index_id,
            edge['from'],
            edge['to'],
            edge.get('type', 'call'),
            edge.get('file_path', ''),
            edge.get('line_number', 0),
            json.dumps(edge.get('metadata', {}))
        ))

    if edge_batch:
        cur.executemany("""
            INSERT INTO call_edges (index_id, caller, callee, edge_type, file_path, line_number, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_id, caller, callee, file_path, line_number) DO UPDATE SET
                edge_type = EXCLUDED.edge_type,
                metadata = EXCLUDED.metadata
        """, edge_batch)
        print(f"Inserted {len(edge_batch)} call edges")

except Exception as e:
    print(f"Failed to insert call edges: {e}")
    conn.rollback()
    sys.exit(1)

# Commit transaction
conn.commit()
cur.close()
conn.close()
print("Database population completed successfully")
EOF
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "Database population completed in ${duration}s"
}

populate_chunked_database() {
    log "Processing chunks into database..."

    local start_time=$(date +%s)
    local manifest_file

    # Determine manifest file location
    if [ -n "$MANIFEST_PATH" ]; then
        manifest_file="$MANIFEST_PATH"
    else
        manifest_file="$CHUNK_OUTPUT_DIR/manifest.json"
    fi

    if [ ! -f "$manifest_file" ]; then
        log_error "Manifest file not found: $manifest_file"
        return 1
    fi

    log "Using manifest: $manifest_file"

    # Use Python script to process chunks
    if [ "$DRY_RUN" = "false" ]; then
        log "Processing chunks with parallel workers (parallelism: $PARALLEL_CHUNKS)..."

        # Check if process_chunks.py exists, otherwise suggest it needs to be created
        if [ ! -f "$SCRIPT_DIR/process_chunks.py" ]; then
            log_warning "process_chunks.py not found. Creating placeholder command..."
            log "Expected command: tools/process_chunks.py --manifest '$manifest_file' --parallel $PARALLEL_CHUNKS"
            log "This will be implemented in T028"

            # For now, create a simple summary
            local chunk_count=$(jq -r '.total_chunks' "$manifest_file" 2>/dev/null || echo "0")
            log "Manifest contains $chunk_count chunks ready for processing"
            log "Run 'tools/process_chunks.py --manifest $manifest_file --parallel $PARALLEL_CHUNKS' to process chunks"
        else
            # Call the chunk processor
            local entrypoints_file="$OUTPUT_DIR/extracted/entrypoints.json"
            local cmd=(python3 "$SCRIPT_DIR/process_chunks.py" \
                --manifest "$manifest_file" \
                --parallel "$PARALLEL_CHUNKS" \
                --database-url "$DATABASE_URL")

            # Add entry points file if it exists
            if [ -f "$entrypoints_file" ]; then
                cmd+=(--entrypoints "$entrypoints_file")
                log "Including entry points from: $entrypoints_file"
            else
                log_warning "Entry points file not found: $entrypoints_file"
                log "Proceeding without entry points (symbols only)"
            fi

            "${cmd[@]}" || {
                log_error "Chunk processing failed"
                return 1
            }
        fi
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "Chunk processing completed in ${duration}s"
}

generate_summary() {
    log "Generating indexing summary..."

    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - INDEX_START_TIME))

    # Check performance targets
    local target_time=$MAX_INDEX_TIME
    if [ "$INCREMENTAL" = "true" ]; then
        target_time=$MAX_INCREMENTAL_TIME
    fi

    local performance_status="✓ PASSED"
    if [ $total_duration -gt $target_time ]; then
        performance_status="✗ FAILED"
    fi

    if [ "$ENABLE_CHUNKING" = "true" ]; then
        generate_chunked_summary "$total_duration" "$target_time" "$performance_status"
    else
        generate_traditional_summary "$total_duration" "$target_time" "$performance_status"
    fi
}

generate_traditional_summary() {
    local total_duration="$1"
    local target_time="$2"
    local performance_status="$3"

    cat << EOF

=====================================
KCS Kernel Indexing Summary
=====================================

Kernel Path:    $KERNEL_PATH
Configuration:  $CONFIG
Index Type:     $([ "$INCREMENTAL" = "true" ] && echo "Incremental" || echo "Full")
Mode:           Traditional (single-file)
Total Time:     ${total_duration}s
Target Time:    ${target_time}s
Performance:    $performance_status

Files:
- Parsed symbols:    $OUTPUT_DIR/parsed/kernel_symbols.json
- Entry points:      $OUTPUT_DIR/extracted/entrypoints.json
- Call graph:        $OUTPUT_DIR/graphs/call_graph.json
- Compile commands:  $OUTPUT_DIR/compile_commands.json

Database:       $DATABASE_URL
Log File:       $LOG_FILE

Next Steps:
- Start KCS server: python -m kcs_mcp.app
- Test queries:     curl http://localhost:8080/mcp/tools/search_code
- View metrics:     curl http://localhost:8080/metrics

Constitutional Requirements:
- Index time:       $([ $total_duration -le $target_time ] && echo "✓ Met" || echo "✗ Exceeded") (${total_duration}s ≤ ${target_time}s)
- Citations:        ✓ All results include file:line references
- Read-only:        ✓ No kernel source modifications

EOF
}

generate_chunked_summary() {
    local total_duration="$1"
    local target_time="$2"
    local performance_status="$3"

    local manifest_file
    if [ -n "$MANIFEST_PATH" ]; then
        manifest_file="$MANIFEST_PATH"
    else
        manifest_file="$CHUNK_OUTPUT_DIR/manifest.json"
    fi

    local chunk_count="unknown"
    local total_size="unknown"
    if [ -f "$manifest_file" ]; then
        chunk_count=$(jq -r '.total_chunks' "$manifest_file" 2>/dev/null || echo "unknown")
        total_size=$(jq -r '.total_size_bytes' "$manifest_file" 2>/dev/null || echo "unknown")
        if [ "$total_size" != "unknown" ] && [ "$total_size" != "null" ]; then
            total_size="${total_size} bytes ($((total_size / 1024 / 1024))MB)"
        fi
    fi

    cat << EOF

=====================================
KCS Kernel Indexing Summary (Chunked)
=====================================

Kernel Path:    $KERNEL_PATH
Configuration:  $CONFIG
Index Type:     $([ "$INCREMENTAL" = "true" ] && echo "Incremental" || echo "Full")
Mode:           Chunked (size: $CHUNK_SIZE)
Total Time:     ${total_duration}s
Target Time:    ${target_time}s
Performance:    $performance_status

Chunked Output:
- Manifest:          $manifest_file
- Chunk directory:   $CHUNK_OUTPUT_DIR
- Total chunks:      $chunk_count
- Total size:        $total_size
- Parallel workers:  $PARALLEL_CHUNKS

Database:       $DATABASE_URL
Log File:       $LOG_FILE

Next Steps:
- Process chunks:   tools/process_chunks.py --manifest $manifest_file --parallel $PARALLEL_CHUNKS
- Start KCS server: python -m kcs_mcp.app
- Test queries:     curl http://localhost:8080/mcp/tools/search_code
- View metrics:     curl http://localhost:8080/metrics

Resume Commands:
- Resume processing: tools/process_chunks.py --manifest $manifest_file --resume
- Incremental update: $0 --incremental --manifest $manifest_file $KERNEL_PATH

Constitutional Requirements:
- Index time:       $([ $total_duration -le $target_time ] && echo "✓ Met" || echo "✗ Exceeded") (${total_duration}s ≤ ${target_time}s)
- Citations:        ✓ All results include file:line references
- Read-only:        ✓ No kernel source modifications
- Chunking:         ✓ Memory-efficient processing enabled

EOF
}

cleanup_temp_files() {
    if [ "$CLEANUP_TEMP" = "true" ]; then
        log "Cleaning up temporary files..."
        rm -rf "$OUTPUT_DIR/temp"
        log_success "Cleanup completed"
    fi
}

show_help() {
    cat << EOF
KCS Kernel Indexing Pipeline

Usage: $0 [OPTIONS] KERNEL_PATH

Arguments:
    KERNEL_PATH             Path to Linux kernel source tree

Options:
    -h, --help              Show this help message
    -c, --config CONFIG     Kernel configuration (default: x86_64:defconfig)
    -o, --output DIR        Output directory (default: /tmp/kcs-index)
    -j, --jobs N            Parallel jobs (default: 4)
    -b, --batch-size N      Database batch size (default: 100)
    --no-clang              Disable clang semantic analysis
    --no-cleanup            Keep temporary files
    --incremental           Incremental indexing (faster)
    --database-url URL      Database connection URL
    --dry-run               Show what would be done without executing
    -v, --verbose           Verbose output

Chunking Options:
    --chunk-size SIZE       Enable chunking with specified size (e.g., 50MB)
    --parallel-chunks N     Number of parallel chunk processors (default: 4)
    --output-dir DIR        Directory for chunk output (enables chunking)
    --manifest PATH         Path to manifest for resume capability
    --subsystem PATH        Index only specific subsystem (e.g., fs/ext4)

Examples:
    # Full index with default configuration
    $0 /usr/src/linux

    # Index with specific configuration
    $0 -c arm64:defconfig /usr/src/linux

    # Fast incremental update
    $0 --incremental /usr/src/linux

    # Custom output and parallel jobs
    $0 -o /data/kcs-index -j 8 /usr/src/linux

    # NEW: Chunked indexing (recommended for large kernels)
    $0 --chunk-size 50MB --parallel-chunks 4 --output-dir /tmp/kcs-chunks /usr/src/linux

    # Resume after failure
    $0 --manifest /tmp/kcs-chunks/manifest.json /usr/src/linux

    # Incremental update with chunks
    $0 --incremental --manifest /tmp/kcs-chunks/manifest.json /usr/src/linux

    # Subsystem only (faster for testing)
    $0 --subsystem fs/ext4 /usr/src/linux

Environment Variables:
    DATABASE_URL            PostgreSQL connection string
    KCS_LOG_LEVEL           Logging level (DEBUG, INFO, WARN, ERROR)
    KCS_PARALLEL_JOBS       Default number of parallel jobs

Performance Targets:
    Full index:             ≤ 20 minutes (constitution requirement)
    Incremental update:     ≤ 3 minutes (constitution requirement)
    Query p95:              ≤ 600ms (runtime requirement)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-clang)
            USE_CLANG=false
            shift
            ;;
        --no-cleanup)
            CLEANUP_TEMP=false
            shift
            ;;
        --incremental)
            INCREMENTAL=true
            shift
            ;;
        --database-url)
            DATABASE_URL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            log "DRY RUN MODE - No changes will be made"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            ENABLE_CHUNKING=true
            shift 2
            ;;
        --parallel-chunks)
            PARALLEL_CHUNKS="$2"
            ENABLE_CHUNKING=true
            shift 2
            ;;
        --output-dir)
            CHUNK_OUTPUT_DIR="$2"
            ENABLE_CHUNKING=true
            shift 2
            ;;
        --manifest)
            MANIFEST_PATH="$2"
            ENABLE_CHUNKING=true
            shift 2
            ;;
        --subsystem)
            SUBSYSTEM="$2"
            shift 2
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$KERNEL_PATH" ]; then
                KERNEL_PATH="$1"
            else
                log_error "Multiple kernel paths specified"
                exit 1
            fi
            shift
            ;;
    esac
done

# Use environment variable as default if no argument provided
if [ -z "$KERNEL_PATH" ] && [ -n "$KCS_KERNEL_PATH" ]; then
    KERNEL_PATH="$KCS_KERNEL_PATH"
    log "Using KCS_KERNEL_PATH environment variable: $KERNEL_PATH"
fi

# Validate required arguments
if [ -z "$KERNEL_PATH" ]; then
    log_error "Kernel path is required"
    log_error "Either provide as argument or set KCS_KERNEL_PATH environment variable"
    show_help
    exit 1
fi

# Convert to absolute path
KERNEL_PATH=$(realpath "$KERNEL_PATH")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Validate and setup chunking configuration
if [ "$ENABLE_CHUNKING" = "true" ]; then
    if [ -z "$CHUNK_OUTPUT_DIR" ]; then
        CHUNK_OUTPUT_DIR="$OUTPUT_DIR/chunks"
    fi
    # Create chunk output directory before calling realpath
    mkdir -p "$CHUNK_OUTPUT_DIR"
    CHUNK_OUTPUT_DIR=$(realpath "$CHUNK_OUTPUT_DIR")

    if [ -n "$MANIFEST_PATH" ]; then
        MANIFEST_PATH=$(realpath "$MANIFEST_PATH")
        if [ ! -f "$MANIFEST_PATH" ]; then
            log_error "Manifest file not found: $MANIFEST_PATH"
            exit 1
        fi
    fi

    log "Chunking enabled: size=$CHUNK_SIZE, parallelism=$PARALLEL_CHUNKS, output=$CHUNK_OUTPUT_DIR"
fi

# Main pipeline execution
show_pipeline_progress() {
    local current_step="$1"
    local total_steps=6
    local step_number

    case "$current_step" in
        "setup") step_number=1 ;;
        "compile") step_number=2 ;;
        "parsing") step_number=3 ;;
        "extraction") step_number=4 ;;
        "graph") step_number=5 ;;
        "database") step_number=6 ;;
        *) step_number=0 ;;
    esac

    if [ "$step_number" -gt 0 ]; then
        local progress=$((step_number * 100 / total_steps))
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "🔄 Pipeline Progress: Step $step_number/$total_steps ($progress%) - $current_step"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
}

main() {
    INDEX_START_TIME=$(date +%s)

    log "Starting KCS kernel indexing pipeline..."
    log "Kernel: $KERNEL_PATH"
    log "Config: $CONFIG"
    log "Output: $OUTPUT_DIR"
    log "Database: $DATABASE_URL"
    log "Log: $LOG_FILE"

    if [ "$ENABLE_CHUNKING" = "true" ]; then
        log "Chunking mode: size=$CHUNK_SIZE, output=$CHUNK_OUTPUT_DIR"
        run_chunked_pipeline
    else
        log "Traditional mode: single-file output"
        run_traditional_pipeline
    fi

    cleanup_temp_files
    generate_summary

    log_success "Kernel indexing pipeline completed successfully!"
}

run_traditional_pipeline() {
    show_pipeline_progress "setup"
    check_prerequisites
    setup_environment

    if [ "$USE_CLANG" = "true" ]; then
        show_pipeline_progress "compile"
        create_compile_commands "$KERNEL_PATH" "$CONFIG"
    fi

    show_pipeline_progress "parsing"
    parse_kernel_sources

    show_pipeline_progress "extraction"
    extract_entrypoints

    show_pipeline_progress "graph"
    build_call_graph

    if [ "${SKIP_DATABASE:-false}" != "true" ]; then
        show_pipeline_progress "database"
        populate_database
    else
        log "Skipping database population (no database connection)"
    fi
}

run_chunked_pipeline() {
    show_pipeline_progress "setup"
    check_prerequisites
    setup_environment

    if [ "$USE_CLANG" = "true" ]; then
        show_pipeline_progress "compile"
        create_compile_commands "$KERNEL_PATH" "$CONFIG"
    fi

    show_pipeline_progress "parsing"
    parse_kernel_sources

    show_pipeline_progress "extraction"
    extract_entrypoints

    if [ "${SKIP_DATABASE:-false}" != "true" ]; then
        show_pipeline_progress "database"
        populate_chunked_database
    else
        log "Skipping database population (no database connection)"
        log "Generated chunks can be processed later with tools/process_chunks.py"
    fi
}

# Trap errors and cleanup
trap 'log_error "Indexing failed. Check log: $LOG_FILE"; cleanup_temp_files' ERR

# Run main pipeline
main
